import os
import sys
import json
import numpy as np
import types

class _FakeLGBM:
    def __init__(self, **kwargs):   pass
    def __setstate__(self, s):      self.__dict__.update(s)
    def predict_proba(self, X):     raise RuntimeError("Use ONNX")
    def predict(self, X):           raise RuntimeError("Use ONNX")
    @property
    def feature_importances_(self): return []


class _AutoMock(types.ModuleType):
    def __getattr__(self, name): return _FakeLGBM


def _make_lgb(name):
    m = _AutoMock(name)
    m.LGBMClassifier = _FakeLGBM
    m.LGBMRegressor  = _FakeLGBM
    m.LGBMRanker     = _FakeLGBM
    m.LGBMModel      = _FakeLGBM
    m.Dataset        = type('Dataset', (), {'__init__': lambda s, *a, **k: None})
    m.train = m.cv   = lambda *a, **k: None
    return m


for _mod in ['lightgbm', 'lightgbm.sklearn', 'lightgbm.basic', 'lightgbm.compat',
             'lightgbm.callback', 'lightgbm.engine', 'lightgbm.plotting', 'lightgbm.dask']:
    sys.modules[_mod] = _make_lgb(_mod)


BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from src.utils import load_config
from src.features import FeatureExtractor
from src.fusion import ScoreFusion


DEEP_FEATURE_NAMES = [
    'deep_feat_mean', 'deep_feat_std', 'deep_feat_max', 'deep_feat_min',
    'deep_temporal_var_mean', 'deep_temporal_var_std',
    'deep_l2_norm_mean', 'deep_l2_norm_std',
    'deep_similarity_mean', 'deep_similarity_std', 'deep_sparsity',
]



_scaler_mean   = None
_scaler_scale  = None
_feature_names = None
_extractor     = None
_fusion        = None
_config        = None
_n_features    = 0
_models_dir    = None
_onnx_session  = None  


def _manual_scale(vector: np.ndarray) -> np.ndarray:
    """Manual StandardScaler"""
    if _scaler_mean is None or _scaler_scale is None:
        raise RuntimeError("Scaler params not loaded")
    
    vector = vector.astype(np.float64)
    scaled = (vector - _scaler_mean) / _scaler_scale
    scaled = np.clip(scaled, -100.0, 100.0)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scaled.astype(np.float32)

def load_models(scaler_path: str, config_path: str, onnx_path: str = None) -> str:
    """Load models và khởi tạo ONNX session trong Python"""
    global _scaler_mean, _scaler_scale, _feature_names
    global _extractor, _fusion, _config, _n_features, _models_dir, _onnx_session

    try:
        _config = load_config(config_path)

        preproc = _config.setdefault('preprocessing', {})
        preproc.setdefault('resize_width',  512)
        preproc.setdefault('resize_height', 288)
        preproc.setdefault('fps',           6)
        preproc.setdefault('max_frames',    1000)

        _config.setdefault('features', {})['use_deep_features'] = False

        _models_dir = os.path.dirname(os.path.abspath(scaler_path))
        _config.setdefault('deep_learning', {})['models_dir'] = _models_dir

        print(f"[DEBUG] Models directory: {_models_dir}")

        model_stem = os.path.splitext(os.path.basename(scaler_path))[0]
        if model_stem.endswith('_scaler'):
            base_stem = model_stem[:-len('_scaler')]
        else:
            base_stem = model_stem

        params_path = os.path.join(_models_dir, f"{base_stem}_scaler_params.json")

        if os.path.exists(params_path):
            print(f"[DEBUG] Loading scaler params from JSON: {params_path}")
            with open(params_path, 'r', encoding='utf-8') as f:
                params = json.load(f)

            _scaler_mean  = np.array(params['mean_'],  dtype=np.float64)
            _scaler_scale = np.array(params['scale_'], dtype=np.float64)
            _n_features   = int(params['n_features'])
            _feature_names = params.get('feature_names')

            print(f"[DEBUG] Loaded JSON scaler params: {_n_features} features")
        else:
            print(f"[WARNING] scaler_params.json not found, falling back to pkl")
            import joblib
            data = joblib.load(scaler_path)
            
            if 'scaler' not in data:
                return 'ERROR: pkl has no "scaler" key'

            sklearn_scaler = data['scaler']
            _feature_names = data.get('feature_names')

            if hasattr(sklearn_scaler, 'n_features_in_'):
                _n_features = int(sklearn_scaler.n_features_in_)
            else:
                _n_features = int(data.get('n_features', 0))

            if hasattr(sklearn_scaler, 'mean_') and hasattr(sklearn_scaler, 'scale_'):
                _scaler_mean  = sklearn_scaler.mean_.astype(np.float64)
                _scaler_scale = sklearn_scaler.scale_.astype(np.float64)
            else:
                return 'ERROR: Scaler not fitted'

        zero_mask = _scaler_scale == 0
        if np.any(zero_mask):
            n_zero = np.sum(zero_mask)
            print(f"[WARNING] {n_zero} features have scale=0, replacing with 1.0")
            _scaler_scale = _scaler_scale.copy()
            _scaler_scale[zero_mask] = 1.0

        _scaler_mean  = np.nan_to_num(_scaler_mean,  nan=0.0, posinf=0.0, neginf=0.0)
        _scaler_scale = np.nan_to_num(_scaler_scale, nan=1.0, posinf=1.0, neginf=1.0)

        if onnx_path and os.path.exists(onnx_path):
            try:
                import onnxruntime as rt
                _onnx_session = rt.InferenceSession(onnx_path)
                print(f"[DEBUG] Loaded ONNX model in Python: {onnx_path}")
                print(f"[DEBUG] ONNX inputs: {[i.name for i in _onnx_session.get_inputs()]}")
                print(f"[DEBUG] ONNX outputs: {[o.name for o in _onnx_session.get_outputs()]}")
            except Exception as e:
                print(f"[WARNING] Could not load ONNX in Python: {e}")
                _onnx_session = None
        else:
            print(f"[WARNING] ONNX path not provided or not found: {onnx_path}")
            _onnx_session = None

        _extractor_temp = FeatureExtractor(_config)
        extractor_names = _extractor_temp.get_feature_names()

        if _feature_names is not None and len(_feature_names) == _n_features:
            print(f"[DEBUG] Using feature_names ({len(_feature_names)} features)")
        else:
            print(f"[WARNING] Rebuilding feature_names from extractor + deep names")
            traditional_names = extractor_names
            all_names = traditional_names + DEEP_FEATURE_NAMES
            if len(all_names) == _n_features:
                _feature_names = all_names
            elif _n_features == 28:
                _feature_names = traditional_names
            else:
                return f'ERROR: Cannot build feature_names: scaler expects={_n_features}'

        _extractor = FeatureExtractor(_config)
        _fusion    = ScoreFusion(_config)

        deep_n = sum(1 for n in _feature_names if n in DEEP_FEATURE_NAMES)
        print(f"[DEBUG] Ready: {_n_features} features ({_n_features - deep_n} trad + {deep_n} deep)")
        return 'OK'

    except Exception as e:
        import traceback
        return f'ERROR: {e}\n{traceback.format_exc()}'



def extract_features(video_path: str, deep_features_json: str = "") -> str:
    if _scaler_mean is None or _extractor is None:
        return json.dumps({'error': 'Model not loaded'})

    try:
        features, metadata = _extractor.extract_from_video(video_path)
        print(f"[DEBUG] Traditional features extracted: {len(features)} keys")

        deep_count = 0
        deep_valid = False

        raw_json = (deep_features_json or "").strip()
        if raw_json and raw_json not in ('{}', 'null', 'None'):
            try:
                deep_feats = json.loads(raw_json)
                if isinstance(deep_feats, dict) and len(deep_feats) > 0:
                    valid_deep = {
                        k: float(v)
                        for k, v in deep_feats.items()
                        if k in DEEP_FEATURE_NAMES
                           and v is not None
                           and not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))
                    }
                    all_zero = all(abs(v) < 1e-10 for v in valid_deep.values())
                    if not all_zero:
                        features.update(valid_deep)
                        deep_valid = True
                        deep_count = len(valid_deep)
                        print(f"[DEBUG] Merged {deep_count} deep features")
            except Exception as e:
                print(f"[WARNING] Parse deep_features_json: {e}")

        for name in DEEP_FEATURE_NAMES:
            if name not in features:
                features[name] = 0.0

        vector = []
        missing = []

        for name in _feature_names:
            val = features.get(name, None)
            if val is None:
                missing.append(name)
                val = 0.0
            if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                val = 0.0
            vector.append(float(val))

        vector = np.array(vector, dtype=np.float64)

        if len(vector) < _n_features:
            vector = np.concatenate([vector, np.zeros(_n_features - len(vector))])
        elif len(vector) > _n_features:
            vector = vector[:_n_features]

        print(f"[DEBUG] Raw vector: shape={vector.shape}, mean={np.mean(vector):.4f}")

        scaled = _manual_scale(vector)
        print(f"[DEBUG] Scaled vector: mean={np.mean(scaled):.4f}, std={np.std(scaled):.4f}")

        prediction = None
        prob_fake = None
        
        if _onnx_session is not None:
            try:
                input_name = _onnx_session.get_inputs()[0].name
                X = scaled.reshape(1, -1).astype(np.float32)
                outputs = _onnx_session.run(None, {input_name: X})
                
                print(f"[DEBUG] ONNX outputs count: {len(outputs)}")
                for i, out in enumerate(outputs):
                    print(f"[DEBUG] Output[{i}]: type={type(out)}, shape={getattr(out, 'shape', 'N/A')}")
                    if hasattr(out, 'flatten'):
                        print(f"[DEBUG] Output[{i}] values: {out.flatten()}")
                
                if len(outputs) >= 1:
                    label_out = outputs[0]
                    if hasattr(label_out, 'flatten'):
                        prediction = int(label_out.flatten()[0])
                    else:
                        prediction = int(label_out[0]) if isinstance(label_out, (list, tuple)) else 0
                
                if len(outputs) >= 2:
                    prob_out = outputs[1]
                    print(f"[DEBUG] Probability output type: {type(prob_out)}")
                    
                    if isinstance(prob_out, np.ndarray):
                        flat = prob_out.flatten()
                        if len(flat) >= 2:
                            prob_fake = float(flat[1])  
                        elif len(flat) == 1:
                            prob_fake = float(flat[0])
                    elif isinstance(prob_out, list) and len(prob_out) > 0:
                        if isinstance(prob_out[0], dict):
                            prob_map = prob_out[0]
                            for key in [1, 1.0, "1", 1]:
                                if key in prob_map:
                                    prob_fake = float(prob_map[key])
                                    break
                            if prob_fake is None:
                                values = [float(v) for v in prob_map.values() if isinstance(v, (int, float, np.number))]
                                if len(values) >= 2:
                                    prob_fake = values[1]
                                elif len(values) == 1:
                                    prob_fake = values[0]
                
                print(f"[DEBUG] ONNX inference: prediction={prediction}, prob_fake={prob_fake}")
                
            except Exception as e:
                print(f"[ERROR] ONNX inference failed: {e}")
                import traceback
                print(traceback.format_exc())
        else:
            print("[WARNING] ONNX session not available in Python")

        artifact = _fusion.compute_artifact_score(features)
        reality  = _fusion.compute_reality_score(features)
        fusion   = _fusion.fuse_scores(artifact, reality, 0.5)
        explain  = _fusion.generate_explanation(features, fusion)

        if prediction is None:
            prediction = 1 if artifact > reality else 0
            print(f"[DEBUG] Using fusion-based prediction: {prediction}")
        
        if prob_fake is None:
            prob_fake = artifact / (artifact + reality + 1e-10)
            prob_fake = prob_fake.coerceIn(0.1, 0.9)
            print(f"[DEBUG] Using fusion-based probability: {prob_fake}")

        result = {
            'vector':            scaled.flatten().tolist(),
            'prediction':        prediction,
            'probability_fake':  prob_fake,
            'fusion_artifact':   float(artifact),
            'fusion_reality':    float(reality),
            'fusion_confidence': fusion['confidence'],
            'explanations':      explain[:5],
            'feature_dim':       len(scaled),
            'debug_info': {
                'n_features_expected':    _n_features,
                'n_deep_features':        deep_count,
                'deep_features_valid':    deep_valid,
                'n_missing':              len(missing),
                'scaled_mean':            float(np.mean(scaled)),
                'scaled_std':             float(np.std(scaled)),
            }
        }

        print(f"[DEBUG] Final: prediction={prediction}, prob_fake={prob_fake:.4f}")

        return json.dumps(result)

    except Exception as e:
        import traceback
        return json.dumps({'error': str(e), 'traceback': traceback.format_exc()})


def analyze_video(video_path: str) -> str:
    return json.dumps({'error': 'Deprecated — use extract_features()'})