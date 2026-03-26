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
from src.features import FeatureExtractor
from src.fusion import ScoreFusion

_scaler        = None   # sklearn StandardScaler
_feature_names = None   # list[str] đúng thứ tự lúc train
_extractor     = None
_fusion        = None
_config        = None
_n_features    = 0      # số chiều scaler expect

# ── Fallback feature-name order (khớp với thứ tự lúc train trên PC) ─────────
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
    # deep (11)
    'deep_feat_mean', 'deep_feat_std', 'deep_feat_max', 'deep_feat_min',
    'deep_temporal_var_mean', 'deep_temporal_var_std',
    'deep_l2_norm_mean', 'deep_l2_norm_std',
    'deep_similarity_mean', 'deep_similarity_std',
    'deep_sparsity',
]
# ────────────────────────────────────────────────────────────────────────────


def load_models(scaler_path: str, config_path: str) -> str:
    """
    Load config, scaler (từ *_scaler.pkl do convert_to_onnx.py tạo),
    extractor, và fusion.

    Android chỉ cần scaler để chuẩn hoá vector trước khi đưa vào ONNX.
    ONNX inference chạy bên Kotlin — Python KHÔNG cần LightGBM model.

    scaler_path: đường dẫn đến file <model>_scaler.pkl
    config_path: đường dẫn đến config.yaml
    """
    global _scaler, _feature_names, _extractor, _fusion, _config, _n_features
    try:
        # ── 1. Load config ────────────────────────────────────────────────
        _config = load_config(config_path)

        # Giảm tải cho mobile
        _config.setdefault('preprocessing', {}).update({
            'max_frames':   30,
            'fps':          3,
            'resize_width': 320,
            'resize_height':180,
        })
        # Tắt deep features — torch không có trên Android
        _config.setdefault('features', {})['use_deep_features'] = False

        # ── 2. Load scaler pkl ────────────────────────────────────────────
        # Format từ convert_to_onnx.py:
        #   { 'scaler': ..., 'calibrator': ..., 'feature_names': [...], 'n_features': int }
        # Cũng hỗ trợ full pkl cũ:
        #   { 'model': ..., 'scaler': ..., 'calibrator': ..., 'feature_names': [...] }
        import joblib
        data = joblib.load(scaler_path)

        if 'scaler' not in data:
            return 'ERROR: file pkl không chứa key "scaler"'

        _scaler     = data['scaler']
        _feature_names = data.get('feature_names')

        # n_features từ scaler (ưu tiên) hoặc field trong pkl
        if hasattr(_scaler, 'n_features_in_'):
            _n_features = int(_scaler.n_features_in_)
        else:
            _n_features = int(data.get('n_features', 0))

        # Fallback feature_names nếu pkl không lưu
        if _feature_names is None:
            if _n_features >= 39:
                _feature_names = _FALLBACK_FEATURE_NAMES_HYBRID
            else:
                _feature_names = _FALLBACK_FEATURE_NAMES_TRADITIONAL

        # Đồng bộ độ dài feature_names với n_features
        if _n_features == 0:
            _n_features = len(_feature_names)

        # ── 3. Khởi tạo extractor & fusion ───────────────────────────────
        _extractor = FeatureExtractor(_config)
        _fusion    = ScoreFusion(_config)

        return 'OK'

    except Exception as e:
        import traceback
        return f'ERROR: {e}\n{traceback.format_exc()}'


def extract_features(video_path: str) -> str:
    """
    Trích xuất features từ video, chuẩn hoá bằng scaler, trả về JSON:
      vector           : list[float]  — đã scale, Kotlin dùng để chạy ONNX
      fusion_artifact  : float
      fusion_reality   : float
      fusion_confidence: str
      explanations     : list[str]
      feature_dim      : int          — để debug
    """
    if _scaler is None or _extractor is None:
        return json.dumps({'error': 'Model chua duoc load — goi load_models() truoc'})

    try:
        # ── Trích xuất features thô ───────────────────────────────────────
        features, metadata = _extractor.extract_from_video(video_path)

        names = _feature_names  # thứ tự cố định từ lúc train

        # Điền 0 cho các feature mà extractor không tạo ra
        # (vd: deep features khi torch không có trên Android)
        for name in names:
            if name not in features:
                features[name] = 0.0

        # Build vector theo đúng thứ tự
        vector = _extractor.features_to_vector(features, names)

        # Đảm bảo đúng số chiều mà scaler expect
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
    """Deprecated — dùng extract_features() thay thế."""
    return json.dumps({'error': 'Dung extract_features thay the'})