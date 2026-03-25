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

# Đăng ký lightgbm và tất cả submodule có thể bị import
for _mod in [
    'lightgbm',
    'lightgbm.sklearn',
    'lightgbm.basic',
    'lightgbm.compat',
    'lightgbm.callback',
    'lightgbm.engine',
    'lightgbm.plotting',
    'lightgbm.dask',
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

def load_models(model_path: str, config_path: str) -> str:
    """
    Load config, extractor, fusion.
    KHÔNG load model pkl/onnx nữa — ONNX chạy bên Kotlin.
    model_path vẫn nhận để lấy feature_names từ scaler pkl.
    """
    global _classifier, _extractor, _fusion, _config
    try:
        _config = load_config(config_path)
        _config.setdefault('preprocessing', {}).update({
            'max_frames': 30, 'target_fps': 3,
            'resize_width': 320, 'resize_height': 180
        })

        # Load classifier CHỈ để lấy scaler + feature_names
        # (model bên trong sẽ không được gọi predict)
        _classifier = VideoClassifier(_config)
        _classifier.load(model_path)

        _extractor = FeatureExtractor(_config)
        _fusion    = ScoreFusion(_config)
        return 'OK'
    except Exception as e:
        return f'ERROR: {e}'


def extract_features(video_path: str) -> str:
    """
    Extract features từ video, trả về JSON chứa:
    - vector: list[float] đã scale, sẵn sàng cho ONNX
    - fusion_artifact: float
    - fusion_reality: float
    - fusion_confidence: str
    - explanations: list[str]
    """
    if not _extractor:
        return json.dumps({'error': 'Model chua duoc load'})
    try:
        features, metadata = _extractor.extract_from_video(video_path)
        names  = _classifier.feature_names or _extractor.get_feature_names()
        vector = _extractor.features_to_vector(features, names)

        # Scale bằng scaler đã load (sklearn StandardScaler)
        scaled = _classifier.scaler.transform(vector.reshape(1, -1))
        vector_list = scaled.flatten().tolist()

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
        })
    except Exception as e:
        return json.dumps({'error': str(e)})


# Giữ lại analyze_video để tương thích nếu cần fallback
def analyze_video(video_path: str) -> str:
    return json.dumps({'error': 'Dung extract_features thay the'})