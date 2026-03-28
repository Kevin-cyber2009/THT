"""
detector.py — Android Chaquopy bridge for AIChecker.

KEY FIX: Uses manual scaling (mean_/scale_ from JSON) instead of sklearn's
StandardScaler.transform() to avoid sklearn 1.5.1 vs 1.1.3 version mismatch.

Pipeline:
  1. Kotlin extracts deep features via ONNX.
  2. Python extracts traditional forensic + reality features.
  3. Python merges both, scales MANUALLY, returns vector to Kotlin.
  4. Kotlin runs the LightGBM ONNX classifier.
"""

import os
import sys
import json
import numpy as np
import types


# ─── LightGBM stub ───────────────────────────────────────────────────────────

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


# ─── Constants ───────────────────────────────────────────────────────────────

DEEP_FEATURE_NAMES = [
    'deep_feat_mean', 'deep_feat_std', 'deep_feat_max', 'deep_feat_min',
    'deep_temporal_var_mean', 'deep_temporal_var_std',
    'deep_l2_norm_mean', 'deep_l2_norm_std',
    'deep_similarity_mean', 'deep_similarity_std',
    'deep_sparsity',
]


# ─── Module-level state ───────────────────────────────────────────────────────

_scaler_mean   = None   # np.ndarray shape (n_features,)
_scaler_scale  = None   # np.ndarray shape (n_features,)
_feature_names = None   # ordered list (39 items)
_extractor     = None
_fusion        = None
_config        = None
_n_features    = 0
_models_dir    = None


def _manual_scale(vector: np.ndarray) -> np.ndarray:
    """
    Manual StandardScaler: x_scaled[i] = (x[i] - mean_[i]) / scale_[i]
    Completely bypasses sklearn version compatibility issues.
    """
    if _scaler_mean is None or _scaler_scale is None:
        raise RuntimeError("Scaler params not loaded")
    scaled = (vector.astype(np.float64) - _scaler_mean) / _scaler_scale
    return scaled.astype(np.float32)


# ─── load_models ─────────────────────────────────────────────────────────────

def load_models(scaler_path: str, config_path: str) -> str:
    global _scaler_mean, _scaler_scale, _feature_names
    global _extractor, _fusion, _config, _n_features, _models_dir

    try:
        _config = load_config(config_path)

        # Preprocessing defaults
        preproc = _config.setdefault('preprocessing', {})
        preproc.setdefault('resize_width',  512)
        preproc.setdefault('resize_height', 288)
        preproc.setdefault('fps',           6)
        preproc.setdefault('max_frames',    1000)

        # Disable Python-side deep learning — Kotlin supplies deep features
        _config.setdefault('features', {})['use_deep_features'] = False

        _models_dir = os.path.dirname(os.path.abspath(scaler_path))
        _config.setdefault('deep_learning', {})['models_dir'] = _models_dir

        print(f"[DEBUG] Models directory: {_models_dir}")

        # ── Try to load scaler params JSON first (preferred) ──────────────────
        model_stem   = os.path.splitext(os.path.basename(scaler_path))[0]
        # Handle "modelname_scaler" → "modelname"
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
            print(f"[DEBUG] Manual scaling: mean[0]={_scaler_mean[0]:.4f}, scale[0]={_scaler_scale[0]:.4f}")

        else:
            # Fallback: load from pkl (may have version warnings)
            print(f"[WARNING] scaler_params.json not found at {params_path}")
            print(f"[WARNING] Falling back to sklearn scaler from pkl")
            print(f"[WARNING] Run save_scaler_params.py on PC to fix this!")

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
                print(f"[DEBUG] Extracted scaler params from pkl: {_n_features} features")
            else:
                return 'ERROR: Scaler not fitted (no mean_/scale_ attributes)'

        # Validate
        if _scaler_mean is None or len(_scaler_mean) != _n_features:
            return f'ERROR: Scaler mean_ length {len(_scaler_mean) if _scaler_mean is not None else 0} != n_features {_n_features}'
        if _scaler_scale is None or len(_scaler_scale) != _n_features:
            return f'ERROR: Scaler scale_ length {len(_scaler_scale) if _scaler_scale is not None else 0} != n_features {_n_features}'

        # Zero scale guard (would cause division by zero)
        zero_mask = _scaler_scale == 0
        if np.any(zero_mask):
            n_zero = np.sum(zero_mask)
            print(f"[WARNING] {n_zero} features have scale=0, replacing with 1.0")
            _scaler_scale = _scaler_scale.copy()
            _scaler_scale[zero_mask] = 1.0

        # NaN/Inf guard
        _scaler_mean  = np.nan_to_num(_scaler_mean,  nan=0.0, posinf=0.0, neginf=0.0)
        _scaler_scale = np.nan_to_num(_scaler_scale, nan=1.0, posinf=1.0, neginf=1.0)

        # ── Build feature names ────────────────────────────────────────────────
        _extractor_temp = FeatureExtractor(_config)
        extractor_names = _extractor_temp.get_feature_names()

        if _feature_names is not None and len(_feature_names) == _n_features:
            print(f"[DEBUG] Using feature_names ({len(_feature_names)} features)")
            deep_missing = [n for n in _feature_names if n not in extractor_names]
            if deep_missing:
                print(f"[INFO] Kotlin supplies: {deep_missing}")
        else:
            print(f"[WARNING] Rebuilding feature_names from extractor + deep names")
            traditional_names = extractor_names  # 28
            all_names = traditional_names + DEEP_FEATURE_NAMES
            if len(all_names) == _n_features:
                _feature_names = all_names
            elif _n_features == 28:
                _feature_names = traditional_names
            else:
                return (f'ERROR: Cannot build feature_names: '
                        f'trad={len(traditional_names)}, deep={len(DEEP_FEATURE_NAMES)}, '
                        f'scaler expects={_n_features}')

        _extractor = FeatureExtractor(_config)
        _fusion    = ScoreFusion(_config)

        deep_n = sum(1 for n in _feature_names if n in DEEP_FEATURE_NAMES)
        print(f"[DEBUG] Ready: {_n_features} features ({_n_features - deep_n} trad + {deep_n} deep)")
        return 'OK'

    except Exception as e:
        import traceback
        return f'ERROR: {e}\n{traceback.format_exc()}'


# ─── extract_features ─────────────────────────────────────────────────────────

def extract_features(video_path: str, deep_features_json: str = "") -> str:
    """
    Extract traditional features, merge Kotlin deep features, scale MANUALLY, return vector.
    """
    if _scaler_mean is None or _extractor is None:
        return json.dumps({'error': 'Model not loaded'})

    try:
        # ── Step 1: Traditional features ──────────────────────────────────────
        features, metadata = _extractor.extract_from_video(video_path)
        print(f"[DEBUG] Traditional features: {len(features)}")

        # ── Step 2: Merge Kotlin deep features ───────────────────────────────
        deep_count  = 0
        deep_valid  = False

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
                    if all_zero:
                        print("[WARNING] Deep features from Kotlin all=0 — ONNX extraction failed")
                    else:
                        features.update(valid_deep)
                        deep_valid = True
                        deep_count = len(valid_deep)
                        print(f"[DEBUG] Merged {deep_count} deep features: "
                              f"mean={valid_deep.get('deep_feat_mean', 0):.4f}")
            except Exception as e:
                print(f"[WARNING] Parse deep_features_json: {e}")

        # Ensure all deep feature slots exist (0.0 if not provided)
        for name in DEEP_FEATURE_NAMES:
            if name not in features:
                features[name] = 0.0

        if not deep_valid:
            print("[WARNING] ⚠ Deep features not available — accuracy may differ from PC")

        # ── Step 3: Build feature vector ─────────────────────────────────────
        vector   = []
        missing  = []

        for name in _feature_names:
            val = features.get(name, None)
            if val is None:
                missing.append(name)
                val = 0.0
            if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                val = 0.0
            vector.append(float(val))

        if missing:
            print(f"[WARNING] Missing features ({len(missing)}): {missing}")

        vector = np.array(vector, dtype=np.float64)

        # Pad or trim to exact expected length
        if len(vector) < _n_features:
            vector = np.concatenate([vector, np.zeros(_n_features - len(vector))])
        elif len(vector) > _n_features:
            vector = vector[:_n_features]

        print(f"[DEBUG] Raw vector: shape={vector.shape}, "
              f"mean={np.mean(vector):.4f}, std={np.std(vector):.4f}")

        # ── Step 4: MANUAL scaling (bypasses sklearn version issue) ───────────
        scaled = _manual_scale(vector)

        # Clamp extreme values (prevents NaN/Inf after scaling)
        scaled = np.clip(scaled, -10.0, 10.0)
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)

        print(f"[DEBUG] Scaled vector: mean={np.mean(scaled):.4f}, std={np.std(scaled):.4f}")
        print(f"[DEBUG] Scaled first 5: {[f'{v:.4f}' for v in scaled[:5]]}")

        # Sanity check: all-zero scaled vector = something is wrong
        if np.all(np.abs(scaled) < 1e-6):
            print("[ERROR] Scaled vector is ALL ZERO — check scaler params!")

        vector_list = scaled.flatten().tolist()

        # ── Step 5: Fusion scores ─────────────────────────────────────────────
        artifact = _fusion.compute_artifact_score(features)
        reality  = _fusion.compute_reality_score(features)
        fusion   = _fusion.fuse_scores(artifact, reality, 0.5)
        explain  = _fusion.generate_explanation(features, fusion)

        result = {
            'vector':            vector_list,
            'fusion_artifact':   float(artifact),
            'fusion_reality':    float(reality),
            'fusion_confidence': fusion['confidence'],
            'explanations':      explain[:5],
            'feature_dim':       len(vector_list),
            'debug_info': {
                'n_features_expected':    _n_features,
                'n_deep_features':        deep_count,
                'deep_features_valid':    deep_valid,
                'n_missing':              len(missing),
                'missing':                missing,
                'scaled_mean':            float(np.mean(scaled)),
                'scaled_std':             float(np.std(scaled)),
            }
        }

        print(f"[DEBUG] artifact={artifact:.4f}, reality={reality:.4f}, "
              f"conf={fusion['confidence']}, deep_valid={deep_valid}")

        return json.dumps(result)

    except Exception as e:
        import traceback
        return json.dumps({'error': str(e), 'traceback': traceback.format_exc()})


def analyze_video(video_path: str) -> str:
    return json.dumps({'error': 'Deprecated — use extract_features()'})