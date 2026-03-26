import os, sys, json
import numpy as np
import types

# ── Chặn import lightgbm + tất cả submodules ────────────────────────────────
class _FakeLGBM:
    """Placeholder — ONNX inference chạy bên Kotlin, không cần predict."""
    def __init__(self, **kwargs): pass
    def __setstate__(self, s):   self.__dict__.update(s)
    def predict_proba(self, X):  raise RuntimeError("Dung ONNX thay the")
    def predict(self, X):        raise RuntimeError("Dung ONNX thay the")
    @property
    def feature_importances_(self): return []

class _AutoMock(types.ModuleType):
    """Module giả: trả về _FakeLGBM cho mọi attribute, tự tạo submodule."""
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
from src.classifier import VideoClassifier
from src.features import FeatureExtractor
from src.fusion import ScoreFusion

_classifier = None
_extractor  = None
_fusion     = None
_config     = None

# ─── FIX: Thứ tự feature names khớp với lúc training trên PC ───────────────
# PC dùng get_feature_names() trả về đúng thứ tự này (không sort alphabetically)
# Android PHẢI dùng cùng thứ tự, lấy từ classifier.feature_names trong pkl
_FALLBACK_FEATURE_NAMES_TRADITIONAL = [
    # forensic (17)
    'fft_mean', 'fft_std', 'fft_max', 'fft_high_freq_energy', 'fft_radial_slope',
    'dct_mean', 'dct_std', 'dct_dc_mean', 'dct_ac_energy',
    'prnu_mean', 'prnu_std', 'prnu_autocorr', 'prnu_temporal_consistency',
    'flow_mean_magnitude', 'flow_std_magnitude', 'flow_smoothness', 'flow_temporal_consistency',
    # reality (11)
    'entropy_mean', 'entropy_std', 'entropy_slope',
    'fractal_dim_mean', 'fractal_dim_std',
    'causal_prediction_error', 'causal_predictability',
    'compression_mean', 'compression_std', 'compression_delta_mean', 'complexity_mean',
]

_FALLBACK_FEATURE_NAMES_HYBRID = _FALLBACK_FEATURE_NAMES_TRADITIONAL + [
    # deep (11) — chỉ dùng khi model được train với deep features
    'deep_feat_mean', 'deep_feat_std', 'deep_feat_max', 'deep_feat_min',
    'deep_temporal_var_mean', 'deep_temporal_var_std',
    'deep_l2_norm_mean', 'deep_l2_norm_std',
    'deep_similarity_mean', 'deep_similarity_std',
    'deep_sparsity',
]
# ────────────────────────────────────────────────────────────────────────────


def load_models(model_path: str, config_path: str) -> str:
    """
    Load config, classifier (chỉ lấy scaler + feature_names), extractor, fusion.
    ONNX inference chạy bên Kotlin.
    """
    global _classifier, _extractor, _fusion, _config
    try:
        _config = load_config(config_path)

        # Giảm max_frames để nhanh hơn trên mobile
        _config.setdefault('preprocessing', {}).update({
            'max_frames': 30,
            'fps': 3,
            'resize_width': 320,
            'resize_height': 180,
        })

        # FIX: Tắt deep features trên Android (không có torch)
        # Model vẫn predict đúng vì ONNX nhận vector đã scale,
        # nhưng ta phải đảm bảo feature_names khớp với pkl
        _config.setdefault('features', {})['use_deep_features'] = False

        # Load classifier CHỈ để lấy scaler + feature_names từ pkl
        _classifier = VideoClassifier(_config)
        _classifier.load(model_path)

        _extractor = FeatureExtractor(_config)
        _fusion    = ScoreFusion(_config)

        # Kiểm tra feature_names có trong pkl không
        if _classifier.feature_names is None:
            # Fallback: dùng thứ tự chuẩn — phải khớp với lúc train
            # Nếu model train với 39 features (hybrid), dùng hybrid list
            # Nếu train với 28 features (traditional), dùng traditional list
            n_features = _classifier.scaler.n_features_in_ if hasattr(_classifier.scaler, 'n_features_in_') else 0
            if n_features == 39 or n_features == 46:
                _classifier.feature_names = _FALLBACK_FEATURE_NAMES_HYBRID
            else:
                _classifier.feature_names = _FALLBACK_FEATURE_NAMES_TRADITIONAL

        return 'OK'
    except Exception as e:
        return f'ERROR: {e}'


def extract_features(video_path: str) -> str:
    """
    Extract features từ video, trả về JSON chứa:
    - vector: list[float] đã scale theo ĐÚNG THỨ TỰ feature_names từ pkl
    - fusion_artifact: float
    - fusion_reality: float
    - fusion_confidence: str
    - explanations: list[str]
    - feature_dim: int (để debug)
    """
    if not _extractor:
        return json.dumps({'error': 'Model chua duoc load'})
    try:
        # Extract tất cả features (traditional only vì torch không có trên Android)
        features, metadata = _extractor.extract_from_video(video_path)

        # FIX CHÍNH: Dùng feature_names từ pkl (thứ tự lúc train)
        # Không sort, không dùng get_feature_names() nếu khác pkl
        names = _classifier.feature_names

        # Kiểm tra xem model có cần deep features không
        # Nếu có thì fill 0 cho những deep features bị thiếu
        n_expected = _classifier.scaler.n_features_in_ if hasattr(_classifier.scaler, 'n_features_in_') else len(names)

        if len(names) > len(features) or n_expected > len(features):
            # Model được train với nhiều features hơn ta có (vd: deep features)
            # Fill 0 cho features bị thiếu
            for name in names:
                if name not in features:
                    features[name] = 0.0

        # Build vector theo đúng thứ tự feature_names từ pkl
        vector = _extractor.features_to_vector(features, names)

        # Đảm bảo đúng số chiều
        if len(vector) != n_expected:
            # Cắt hoặc pad
            if len(vector) < n_expected:
                vector = np.concatenate([vector, np.zeros(n_expected - len(vector), dtype=np.float32)])
            else:
                vector = vector[:n_expected]

        # Scale bằng scaler đã train (sklearn StandardScaler)
        scaled = _classifier.scaler.transform(vector.reshape(1, -1))
        vector_list = scaled.flatten().tolist()

        # Tính fusion scores
        artifact   = _fusion.compute_artifact_score(features)
        reality    = _fusion.compute_reality_score(features)
        fusion     = _fusion.fuse_scores(artifact, reality, 0.5)
        explain    = _fusion.generate_explanation(features, fusion)

        return json.dumps({
            'vector':             vector_list,       # Kotlin dùng để chạy ONNX
            'fusion_artifact':    float(artifact),
            'fusion_reality':     float(reality),
            'fusion_confidence':  fusion['confidence'],
            'explanations':       explain[:5],
            'feature_dim':        len(vector_list),  # debug
        })
    except Exception as e:
        import traceback
        return json.dumps({'error': str(e), 'traceback': traceback.format_exc()})


def analyze_video(video_path: str) -> str:
    return json.dumps({'error': 'Dung extract_features thay the'})