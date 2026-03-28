"""
detector.py — Android Chaquopy bridge for AIChecker.

Pipeline:
  1. Kotlin extracts deep features via ONNX (Kotlin ONNX runtime).
  2. Python extracts traditional forensic + reality features.
  3. Python merges both, scales, returns feature vector to Kotlin.
  4. Kotlin runs the LightGBM ONNX classifier.
"""

import os
import sys
import json
import numpy as np
import types


# ─────────────────────────────────────────────────────────────────────────────
#  LightGBM stub — ONNX inference runs in Kotlin, not needed in Python
# ─────────────────────────────────────────────────────────────────────────────

class _FakeLGBM:
    """Placeholder — ONNX inference runs in Kotlin, not needed here."""
    def __init__(self, **kwargs):      pass
    def __setstate__(self, s):         self.__dict__.update(s)
    def predict_proba(self, X):        raise RuntimeError("Use ONNX instead")
    def predict(self, X):              raise RuntimeError("Use ONNX instead")
    @property
    def feature_importances_(self):    return []


class _AutoMock(types.ModuleType):
    def __getattr__(self, name):
        return _FakeLGBM


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


# ─────────────────────────────────────────────────────────────────────────────
#  Canonical deep feature names — MUST match training order exactly
# ─────────────────────────────────────────────────────────────────────────────

DEEP_FEATURE_NAMES = [
    'deep_feat_mean', 'deep_feat_std', 'deep_feat_max', 'deep_feat_min',
    'deep_temporal_var_mean', 'deep_temporal_var_std',
    'deep_l2_norm_mean', 'deep_l2_norm_std',
    'deep_similarity_mean', 'deep_similarity_std',
    'deep_sparsity',
]


# ─────────────────────────────────────────────────────────────────────────────
#  Module-level state
# ─────────────────────────────────────────────────────────────────────────────

_scaler        = None
_feature_names = None   # ordered list from scaler (39 items)
_extractor     = None
_fusion        = None
_config        = None
_n_features    = 0      # expected feature vector length (39)


# ─────────────────────────────────────────────────────────────────────────────
#  load_models
# ─────────────────────────────────────────────────────────────────────────────

def load_models(scaler_path: str, config_path: str) -> str:
    global _scaler, _feature_names, _extractor, _fusion, _config, _n_features
    try:
        _config = load_config(config_path)

        # Preprocessing defaults
        preproc = _config.setdefault('preprocessing', {})
        preproc.setdefault('resize_width',  512)
        preproc.setdefault('resize_height', 288)
        preproc.setdefault('fps',           6)
        preproc.setdefault('max_frames',    1000)

        # Disable Python-side deep learning (Kotlin handles it)
        features_config = _config.setdefault('features', {})
        features_config['use_deep_features'] = False   # ← Kotlin supplies deep features

        # Pass models directory so deep_features_onnx can find files if ever needed
        models_dir = os.path.dirname(os.path.abspath(scaler_path))
        dl_config  = _config.setdefault('deep_learning', {})
        dl_config['models_dir'] = models_dir

        print(f"[DEBUG] Models directory: {models_dir}")

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

        # Build extractor (traditional only — deep comes from Kotlin)
        _extractor_temp  = FeatureExtractor(_config)
        extractor_names  = _extractor_temp.get_feature_names()

        print(f"[DEBUG] Scaler expects {_n_features} features")
        print(f"[DEBUG] Scaler feature_names: {_feature_names[:5] if _feature_names else None}...")
        print(f"[DEBUG] Extractor provides {len(extractor_names)} traditional features")

        if _feature_names is not None and len(_feature_names) == _n_features:
            print(f"[DEBUG] Using feature_names from scaler ({len(_feature_names)} features)")
            missing_in_extractor = [n for n in _feature_names if n not in extractor_names]
            if missing_in_extractor:
                print(f"[INFO] Features supplied by Kotlin ONNX: {missing_in_extractor}")
        else:
            print(f"[WARNING] Scaler feature_names invalid, rebuilding from extractor + deep names")
            traditional_names = extractor_names  # 28 names
            all_names = traditional_names + DEEP_FEATURE_NAMES
            if len(all_names) == _n_features:
                _feature_names = all_names
            elif _n_features == 28:
                _feature_names = traditional_names
            else:
                return (f'ERROR: Feature count mismatch: '
                        f'extractor={len(extractor_names)}, deep={len(DEEP_FEATURE_NAMES)}, '
                        f'scaler expects={_n_features}')

        _extractor = FeatureExtractor(_config)
        _fusion    = ScoreFusion(_config)

        final_names = _extractor.get_feature_names()
        deep_count  = sum(1 for n in _feature_names if n in DEEP_FEATURE_NAMES)
        print(f"[DEBUG] Traditional extractor: {len(final_names)} features")
        print(f"[DEBUG] Total expected: {_n_features} ({_n_features - deep_count} trad + {deep_count} deep)")

        return 'OK'

    except Exception as e:
        import traceback
        return f'ERROR: {e}\n{traceback.format_exc()}'


# ─────────────────────────────────────────────────────────────────────────────
#  extract_features
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(video_path: str, deep_features_json: str = "") -> str:
    """
    Extract traditional features from video, merge pre-computed deep features
    from Kotlin ONNX runtime, apply scaler, return feature vector + fusion scores.

    Args:
        video_path:          Path to the video file.
        deep_features_json:  JSON string {"deep_feat_mean": 0.123, ...} from Kotlin.
                             If "{}" or empty, deep features default to 0.0.
    """
    if _scaler is None or _extractor is None:
        return json.dumps({'error': 'Model not loaded — call load_models() first'})

    try:
        # ── Step 1: Traditional forensic + reality features ───────────────────
        features, metadata = _extractor.extract_from_video(video_path)
        print(f"[DEBUG] Extracted {len(features)} traditional features")

        # ── Step 2: Merge Kotlin deep features ───────────────────────────────
        deep_feat_count  = 0
        deep_feats_valid = False

        if deep_features_json and deep_features_json.strip() not in ('', '{}', 'null', 'None'):
            try:
                deep_feats = json.loads(deep_features_json)
                if isinstance(deep_feats, dict) and len(deep_feats) > 0:
                    # Validate: only accept known deep_* keys
                    valid_deep = {
                        k: float(v)
                        for k, v in deep_feats.items()
                        if k in DEEP_FEATURE_NAMES and v is not None
                    }

                    # Check they're not all zero (zeros mean extraction failed)
                    all_zero = all(abs(v) < 1e-10 for v in valid_deep.values())
                    if all_zero:
                        print("[WARNING] Deep features from Kotlin are ALL ZERO — ignoring")
                        print("[WARNING] This means ONNX IR version mismatch. Re-export ONNX models!")
                        # Fill with zeros explicitly so scaler can still work
                        for name in DEEP_FEATURE_NAMES:
                            features[name] = 0.0
                        deep_feat_count = len(DEEP_FEATURE_NAMES)
                    else:
                        features.update(valid_deep)
                        # Fill any missing deep features with 0
                        for name in DEEP_FEATURE_NAMES:
                            if name not in features:
                                features[name] = 0.0
                        deep_feat_count  = len(valid_deep)
                        deep_feats_valid = True
                        print(f"[DEBUG] Merged {deep_feat_count} valid deep features from Kotlin")
                        print(f"[DEBUG] Sample: mean={valid_deep.get('deep_feat_mean', 0):.4f}, "
                              f"std={valid_deep.get('deep_feat_std', 0):.4f}")
            except Exception as e:
                print(f"[WARNING] Failed to parse deep_features_json: {e}")
                for name in DEEP_FEATURE_NAMES:
                    features[name] = 0.0
                deep_feat_count = len(DEEP_FEATURE_NAMES)
        else:
            print("[DEBUG] No deep features from Kotlin — setting all deep_* to 0.0")
            print("[WARNING] ⚠ Deep features = 0. Accuracy will differ from PC app!")
            print("[WARNING]   Fix: re-export ONNX with export_deep_features_onnx.py (opset=11)")
            for name in DEEP_FEATURE_NAMES:
                features[name] = 0.0
            deep_feat_count = len(DEEP_FEATURE_NAMES)

        print(f"[DEBUG] Total features after merge: {len(features)}")

        # ── Step 3: Build feature vector in EXACT order of _feature_names ─────
        vector           = []
        missing_features = []

        for name in _feature_names:
            value = features.get(name, None)
            if value is None:
                missing_features.append(name)
                value = 0.0
            if np.isnan(value) or np.isinf(value):
                value = 0.0
            vector.append(float(value))

        if missing_features:
            print(f"[WARNING] Missing features ({len(missing_features)}): {missing_features}")

        vector = np.array(vector, dtype=np.float64)

        # Ensure correct length
        if len(vector) != _n_features:
            print(f"[ERROR] Vector length mismatch: {len(vector)} vs expected {_n_features}")
            if len(vector) < _n_features:
                vector = np.concatenate([
                    vector,
                    np.zeros(_n_features - len(vector), dtype=np.float64)
                ])
            else:
                vector = vector[:_n_features]

        print(f"[DEBUG] Feature vector shape: {vector.shape}")
        print(f"[DEBUG] Stats: mean={np.mean(vector):.4f}, std={np.std(vector):.4f}")

        # ── Step 4: Apply scaler ───────────────────────────────────────────────
        try:
            scaled = _scaler.transform(vector.reshape(1, -1))
            print(f"[DEBUG] Scaled stats: mean={np.mean(scaled):.4f}, std={np.std(scaled):.4f}")
        except Exception as e:
            print(f"[ERROR] Scaler transform failed: {e}")
            return json.dumps({'error': f'Scaler transform failed: {e}'})

        vector_list = scaled.flatten().tolist()

        # ── Step 5: Compute fusion scores ──────────────────────────────────────
        artifact = _fusion.compute_artifact_score(features)
        reality  = _fusion.compute_reality_score(features)
        fusion   = _fusion.fuse_scores(artifact, reality, 0.5)
        explain  = _fusion.generate_explanation(features, fusion)

        # ── Build result ───────────────────────────────────────────────────────
        result = {
            'vector':            vector_list,
            'fusion_artifact':   float(artifact),
            'fusion_reality':    float(reality),
            'fusion_confidence': fusion['confidence'],
            'explanations':      explain[:5],
            'feature_dim':       len(vector_list),
            'debug_info': {
                'n_features_expected':    _n_features,
                'n_traditional_features': len(features) - deep_feat_count,
                'n_deep_features':        deep_feat_count,
                'deep_features_valid':    deep_feats_valid,
                'n_missing_features':     len(missing_features),
                'missing_features':       missing_features,
            }
        }

        print(f"[DEBUG] Result: artifact={artifact:.4f}, reality={reality:.4f}, "
              f"confidence={fusion['confidence']}, deep_valid={deep_feats_valid}")

        return json.dumps(result)

    except Exception as e:
        import traceback
        return json.dumps({'error': str(e), 'traceback': traceback.format_exc()})


def analyze_video(video_path: str) -> str:
    """Deprecated — use extract_features() instead."""
    return json.dumps({'error': 'Use extract_features() instead'})