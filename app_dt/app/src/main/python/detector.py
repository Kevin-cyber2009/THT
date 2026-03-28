import os, sys, json
import numpy as np
import types

class _FakeLGBM:
    def __init__(self, **kwargs): pass
    def __setstate__(self, s): self.__dict__.update(s)
    def predict_proba(self, X): raise RuntimeError("Use ONNX instead")
    def predict(self, X):       raise RuntimeError("Use ONNX instead")
    @property
    def feature_importances_(self): return []

class _AutoMock(types.ModuleType):
    def __getattr__(self, name): return _FakeLGBM

def _make_lgb_module(name):
    m = _AutoMock(name)
    m.LGBMClassifier = _FakeLGBM
    m.LGBMRegressor  = _FakeLGBM
    m.LGBMRanker     = _FakeLGBM
    m.LGBMModel      = _FakeLGBM
    m.Dataset        = type('Dataset', (), {'__init__': lambda s, *a, **k: None})
    m.train          = lambda *a, **k: None
    m.cv             = lambda *a, **k: None
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

_scaler               = None
_feature_names        = None
_extractor            = None
_fusion               = None
_config               = None
_n_features           = 0
_ignore_deep_features = True   

_DEEP_NAMES = frozenset([
    'deep_feat_mean', 'deep_feat_std', 'deep_feat_max', 'deep_feat_min',
    'deep_temporal_var_mean', 'deep_temporal_var_std',
    'deep_l2_norm_mean', 'deep_l2_norm_std',
    'deep_similarity_mean', 'deep_similarity_std', 'deep_sparsity',
])


def load_models(scaler_path: str, config_path: str) -> str:
    global _scaler, _feature_names, _extractor, _fusion, _config
    global _n_features, _ignore_deep_features

    try:
        _config = load_config(config_path)

        preproc = _config.setdefault('preprocessing', {})
        preproc.setdefault('resize_width',  512)
        preproc.setdefault('resize_height', 288)
        preproc.setdefault('fps',           6)
        preproc.setdefault('max_frames',    1000)

 
        _config.setdefault('features', {})['use_deep_features'] = False

        models_dir = os.path.dirname(os.path.abspath(scaler_path))
        _config.setdefault('deep_learning', {})['models_dir'] = models_dir

        import joblib
        data = joblib.load(scaler_path)

        if 'scaler' not in data:
            return 'ERROR: pkl does not contain key "scaler"'

        _scaler        = data['scaler']
        _feature_names = data.get('feature_names')
        _n_features    = (
            int(_scaler.n_features_in_) if hasattr(_scaler, 'n_features_in_')
            else int(data.get('n_features', 0))
        )

        _ignore_deep_features = not any(
            n in _DEEP_NAMES for n in (_feature_names or [])
        )

        print(f"[detector] n_features={_n_features}, ignore_deep={_ignore_deep_features}")

        _extractor = FeatureExtractor(_config)
        _fusion    = ScoreFusion(_config)

        trad_names = _extractor.get_feature_names()   

        if _feature_names is None or len(_feature_names) != _n_features:
            if _ignore_deep_features:
                _feature_names = trad_names
            else:
                _feature_names = trad_names + sorted(_DEEP_NAMES - set(trad_names))
            if len(_feature_names) != _n_features:
                return (f'ERROR: feature count mismatch '
                        f'built={len(_feature_names)} scaler={_n_features}')

        print(f"[detector] feature_names[0:5]={_feature_names[:5]}")
        return 'OK'

    except Exception as e:
        import traceback
        return f'ERROR: {e}\n{traceback.format_exc()}'


def _neutral_value_for(idx: int) -> float:
    if hasattr(_scaler, 'mean_') and idx < len(_scaler.mean_):
        return float(_scaler.mean_[idx])
    return 0.0


def extract_features(video_path: str, deep_json: str = '{}') -> str:
    if _scaler is None or _extractor is None:
        return json.dumps({'error': 'Model not loaded — call load_models() first'})

    try:
        features, metadata = _extractor.extract_from_video(video_path)

        deep_features_valid = False

        if not _ignore_deep_features:
            injected = {}
            if deep_json and deep_json not in ('{}', '', 'null'):
                try:
                    parsed   = json.loads(deep_json)
                    injected = {k: float(v) for k, v in parsed.items()
                                if k in _DEEP_NAMES}
                except Exception as e:
                    print(f"[detector] deep_json parse error: {e}")

            if injected:
                features.update(injected)
                deep_features_valid = True
                print(f"[detector] Injected {len(injected)} Kotlin deep features")
            else:
                print("[detector] No Kotlin deep features — neutralising with scaler means")
                for i, name in enumerate(_feature_names):
                    if name in _DEEP_NAMES:
                        features[name] = _neutral_value_for(i)

        vector  = []
        missing = []
        for i, name in enumerate(_feature_names):
            value = features.get(name)
            if value is None:
                missing.append(name)
                value = _neutral_value_for(i)   
            if np.isnan(value) or np.isinf(value):
                value = _neutral_value_for(i)
            vector.append(float(value))

        vector = np.array(vector, dtype=np.float64)

        if len(vector) < _n_features:
            pad = np.array([_neutral_value_for(i)
                            for i in range(len(vector), _n_features)])
            vector = np.concatenate([vector, pad])
        elif len(vector) > _n_features:
            vector = vector[:_n_features]

        print(f"[detector] vector shape={vector.shape}, "
              f"mean={np.mean(vector):.4f}, std={np.std(vector):.4f}")

        scaled = _scaler.transform(vector.reshape(1, -1))

        print(f"[detector] scaled mean={np.mean(scaled):.4f}, "
              f"std={np.std(scaled):.4f}")

        artifact = _fusion.compute_artifact_score(features)
        reality  = _fusion.compute_reality_score(features)
        fusion   = _fusion.fuse_scores(artifact, reality, 0.5)
        explain  = _fusion.generate_explanation(features, fusion)

        return json.dumps({
            'vector':            scaled.flatten().tolist(),
            'fusion_artifact':   float(artifact),
            'fusion_reality':    float(reality),
            'fusion_confidence': fusion['confidence'],
            'explanations':      explain[:5],
            'feature_dim':       int(len(scaled.flatten())),
            'debug_info': {
                'n_features_expected':   _n_features,
                'n_features_extracted':  len(features),
                'n_missing':             len(missing),
                'missing':               missing[:10],
                'deep_features_ignored': bool(_ignore_deep_features),
                'deep_features_valid':   deep_features_valid,
                'scaled_mean':           float(np.mean(scaled)),
                'scaled_std':            float(np.std(scaled)),
            }
        })

    except Exception as e:
        import traceback
        return json.dumps({
            'error':     str(e),
            'traceback': traceback.format_exc()
        })


def analyze_video(video_path: str) -> str:
    """Deprecated – use extract_features() instead."""
    return json.dumps({'error': 'Use extract_features() instead'})