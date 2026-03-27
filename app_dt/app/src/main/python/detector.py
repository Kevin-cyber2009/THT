import os, sys, json
import numpy as np
import types

class _FakeLGBM:
    """Placeholder — ONNX inference runs in Kotlin, not needed here."""
    def __init__(self, **kwargs): pass
    def __setstate__(self, s):   self.__dict__.update(s)
    def predict_proba(self, X):  raise RuntimeError("Use ONNX instead")
    def predict(self, X):        raise RuntimeError("Use ONNX instead")
    @property
    def feature_importances_(self): return []

class _AutoMock(types.ModuleType):
    def __getattr__(self, name):
        return _FakeLGBM

def _make_lgb_module(name):
    m = _AutoMock(name)
    m.LGBMClassifier = _FakeLGBM
    m.LGBMRegressor  = _FakeLGBM
    m.LGBMRanker     = _FakeLGBM
    m.LGBMModel      = _FakeLGBM
    m.Dataset        = type('Dataset', (), {'__init__': lambda s,*a,**k: None})
    m.train          = lambda *a,**k: None
    m.cv             = lambda *a,**k: None
    return m

for _mod in [
    'lightgbm', 'lightgbm.sklearn', 'lightgbm.basic', 'lightgbm.compat',
    'lightgbm.callback', 'lightgbm.engine', 'lightgbm.plotting', 'lightgbm.dask',
]:
    sys.modules[_mod] = _make_lgb_module(_mod)

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from src.utils import load_config
from src.features import FeatureExtractor
from src.fusion import ScoreFusion

_scaler        = None
_feature_names = None
_extractor     = None
_fusion        = None
_config        = None
_n_features    = 0


def load_models(scaler_path: str, config_path: str) -> str:
    global _scaler, _feature_names, _extractor, _fusion, _config, _n_features
    try:
        _config = load_config(config_path)

        preproc = _config.setdefault('preprocessing', {})
        

        preproc['resize_width'] = preproc.get('resize_width', 512)
        preproc['resize_height'] = preproc.get('resize_height', 288)
        preproc['fps'] = preproc.get('fps', 6)
        

        preproc['max_frames'] = preproc.get('max_frames', 1000)


        features_config = _config.setdefault('features', {})
        features_config['use_deep_features'] = features_config.get('use_deep_features', True)

        import joblib
        data = joblib.load(scaler_path)

        if 'scaler' not in data:
            return 'ERROR: pkl file does not contain key "scaler"'

        _scaler        = data['scaler']
        _feature_names = data.get('feature_names')
        
        if hasattr(_scaler, 'n_features_in_'):
            _n_features = int(_scaler.n_features_in_)
        else:
            _n_features = int(data.get('n_features', 0))

        _extractor_temp = FeatureExtractor(_config)
        
        if _feature_names is None or len(_feature_names) != _n_features:

            _feature_names = _extractor_temp.get_feature_names()
            
            if len(_feature_names) > _n_features:
                _feature_names = _feature_names[:_n_features]
            elif len(_feature_names) < _n_features:
                _feature_names = _feature_names + [f'feature_{i}' for i in range(len(_feature_names), _n_features)]
        else:
            extractor_names = _extractor_temp.get_feature_names()
            if len(extractor_names) == _n_features:
                if extractor_names != _feature_names:
                    print(f"WARNING: Model feature_names differ from FeatureExtractor. Using model's feature_names.")

        if _n_features == 0:
            _n_features = len(_feature_names)

        _extractor = FeatureExtractor(_config)
        _fusion    = ScoreFusion(_config)

        return 'OK'

    except Exception as e:
        import traceback
        return f'ERROR: {e}\n{traceback.format_exc()}'


def extract_features(video_path: str) -> str:
    """
    Extract features from video, scale using the saved scaler, return JSON:
      vector           : list[float]  — scaled, ready for ONNX in Kotlin
      fusion_artifact  : float
      fusion_reality   : float
      fusion_confidence: str
      explanations     : list[str]
      feature_dim      : int
    """
    if _scaler is None or _extractor is None:
        return json.dumps({'error': 'Model not loaded — call load_models() first'})

    try:
        features, metadata = _extractor.extract_from_video(video_path)

        vector = []
        for name in _feature_names:
            value = features.get(name, 0.0)
            
            if np.isnan(value) or np.isinf(value):
                value = 0.0
            
            vector.append(value)

        vector = np.array(vector, dtype=np.float32)

        if len(vector) != _n_features:
            if len(vector) < _n_features:
                vector = np.concatenate([
                    vector,
                    np.zeros(_n_features - len(vector), dtype=np.float32)
                ])
            elif len(vector) > _n_features:
                vector = vector[:_n_features]

        scaled      = _scaler.transform(vector.reshape(1, -1))
        vector_list = scaled.flatten().tolist()

        artifact  = _fusion.compute_artifact_score(features)
        reality   = _fusion.compute_reality_score(features)
        fusion    = _fusion.fuse_scores(artifact, reality, 0.5)
        explain   = _fusion.generate_explanation(features, fusion)

        return json.dumps({
            'vector':             vector_list,
            'fusion_artifact':    float(artifact),
            'fusion_reality':     float(reality),
            'fusion_confidence':  fusion['confidence'],
            'explanations':       explain[:5],
            'feature_dim':        len(vector_list),
            'debug_n_features':   _n_features,
            'debug_vector_len':   len(vector),
        })

    except Exception as e:
        import traceback
        return json.dumps({'error': str(e), 'traceback': traceback.format_exc()})


def analyze_video(video_path: str) -> str:
    """Deprecated — use extract_features() instead."""
    return json.dumps({'error': 'Use extract_features() instead'})