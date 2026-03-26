import os, sys, json
import numpy as np
import types

# ── Block lightgbm + all submodules ─────────────────────────────────────────
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
# ────────────────────────────────────────────────────────────────────────────

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

# ── Fallback feature-name order (must match training order) ──────────────────
_FALLBACK_FEATURE_NAMES_TRADITIONAL = [
    'fft_mean', 'fft_std', 'fft_max', 'fft_high_freq_energy', 'fft_radial_slope',
    'dct_mean', 'dct_std', 'dct_dc_mean', 'dct_ac_energy',
    'prnu_mean', 'prnu_std', 'prnu_autocorr', 'prnu_temporal_consistency',
    'flow_mean_magnitude', 'flow_std_magnitude', 'flow_smoothness', 'flow_temporal_consistency',
    'entropy_mean', 'entropy_std', 'entropy_slope',
    'fractal_dim_mean', 'fractal_dim_std',
    'causal_prediction_error', 'causal_predictability',
    'compression_mean', 'compression_std', 'compression_delta_mean', 'complexity_mean',
]

_FALLBACK_FEATURE_NAMES_HYBRID = _FALLBACK_FEATURE_NAMES_TRADITIONAL + [
    'deep_feat_mean', 'deep_feat_std', 'deep_feat_max', 'deep_feat_min',
    'deep_temporal_var_mean', 'deep_temporal_var_std',
    'deep_l2_norm_mean', 'deep_l2_norm_std',
    'deep_similarity_mean', 'deep_similarity_std',
    'deep_sparsity',
]
# ────────────────────────────────────────────────────────────────────────────


def load_models(scaler_path: str, config_path: str) -> str:
    """
    Load config, scaler, feature extractor, and fusion engine.

    IMPORTANT: preprocessing parameters (resize_width, resize_height, fps)
    are kept identical to training config to avoid feature distribution shift.
    Only max_frames is capped for mobile performance.
    """
    global _scaler, _feature_names, _extractor, _fusion, _config, _n_features
    try:
        # ── 1. Load config ────────────────────────────────────────────────
        _config = load_config(config_path)

        # FIX: Do NOT override resize_width, resize_height, or fps.
        # These MUST match training config (512x288, fps=6).
        # Changing spatial resolution shifts the entire frequency domain
        # (FFT radial slope, DCT energy, optical flow magnitudes), making
        # the model predict on out-of-distribution inputs → wrong results.
        #
        # Only cap max_frames for mobile performance. Using 100 frames
        # gives good feature stability while keeping inference time acceptable.
        preproc = _config.setdefault('preprocessing', {})
        if preproc.get('max_frames', 1000) > 100:
            preproc['max_frames'] = 100

        # Disable deep features — PyTorch is not available on Android.
        # Deep feature names in the vector will be filled with 0.0, which
        # is consistent with how the model was trained (key-name mismatch
        # between EnsembleDeepExtractor output keys and get_feature_names()
        # means deep features were effectively 0 during training too).
        _config.setdefault('features', {})['use_deep_features'] = False

        # ── 2. Load scaler pkl ────────────────────────────────────────────
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

        if _feature_names is None:
            if _n_features >= 39:
                _feature_names = _FALLBACK_FEATURE_NAMES_HYBRID
            else:
                _feature_names = _FALLBACK_FEATURE_NAMES_TRADITIONAL

        if _n_features == 0:
            _n_features = len(_feature_names)

        # ── 3. Init extractor & fusion ────────────────────────────────────
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
        # ── Extract raw features ──────────────────────────────────────────
        features, metadata = _extractor.extract_from_video(video_path)

        names = _feature_names

        # Fill zeros for features the extractor did not produce
        # (e.g. deep features when PyTorch is unavailable)
        for name in names:
            if name not in features:
                features[name] = 0.0

        # Build ordered vector matching training feature order
        vector = _extractor.features_to_vector(features, names)

        # Pad or truncate to exact scaler dimension
        if len(vector) < _n_features:
            vector = np.concatenate([
                vector,
                np.zeros(_n_features - len(vector), dtype=np.float32)
            ])
        elif len(vector) > _n_features:
            vector = vector[:_n_features]

        # ── Scale ─────────────────────────────────────────────────────────
        scaled      = _scaler.transform(vector.reshape(1, -1))
        vector_list = scaled.flatten().tolist()

        # ── Fusion scores & explanations ──────────────────────────────────
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
        })

    except Exception as e:
        import traceback
        return json.dumps({'error': str(e), 'traceback': traceback.format_exc()})


def analyze_video(video_path: str) -> str:
    """Deprecated — use extract_features() instead."""
    return json.dumps({'error': 'Use extract_features() instead'})