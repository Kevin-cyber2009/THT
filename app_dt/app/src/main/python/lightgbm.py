# app/src/main/python/lightgbm.py
# Stub thay thế lightgbm trên Android.
# classifier.py import lightgbm as lgb → load file này thay vì thư viện thật.
# Khi load() pkl, model bên trong là LGBMClassifier (object cũ).
# Stub này thay thế predict_proba bằng ONNX runtime tự động.

import numpy as np
import os

# ── Fake LGBMClassifier ───────────────────────────────────────────────────────

class LGBMClassifier:
    """
    Fake LGBMClassifier:
    - Được joblib.load() deserialize từ alpha.pkl mà không lỗi
    - Khi predict_proba() được gọi, tự tìm file .onnx tương ứng và chạy
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self._onnx_session  = None
        self._onnx_input    = "float_input"
        self._model_path    = None   # được set bởi VideoClassifier.load()

    # joblib cần __reduce__ để deserialize đúng
    def __setstate__(self, state):
        self.__dict__.update(state)
        self._onnx_session = None
        self._onnx_input   = "float_input"
        self._model_path   = None

    def _ensure_onnx(self):
        """Load ONNX session lần đầu tiên khi cần predict."""
        if self._onnx_session is not None:
            return

        # Tìm file .onnx: thay đuôi .pkl → .onnx từ path đã biết
        onnx_path = None
        if self._model_path:
            candidate = os.path.splitext(self._model_path)[0] + ".onnx"
            if os.path.exists(candidate):
                onnx_path = candidate

        # Fallback: tìm trong cùng thư mục với pkl
        if onnx_path is None:
            raise RuntimeError(
                "Không tìm thấy file .onnx tương ứng.\n"
                "Hãy chạy convert_to_onnx.py trên PC và copy "
                "alpha.onnx vào assets/models/"
            )

        import onnxruntime as rt
        self._onnx_session = rt.InferenceSession(onnx_path)
        self._onnx_input   = self._onnx_session.get_inputs()[0].name

    def predict_proba(self, X):
        self._ensure_onnx()
        X32     = np.array(X, dtype=np.float32)
        outputs = self._onnx_session.run(None, {self._onnx_input: X32})
        # outputs[1] = list of dicts {0: p_real, 1: p_fake}
        proba_list = outputs[1]
        p_fake  = np.array([d.get(1, d.get(1.0, 0.0)) for d in proba_list])
        p_real  = 1.0 - p_fake
        return np.column_stack([p_real, p_fake])

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    # Các thuộc tính LightGBM mà classifier.py có thể dùng
    @property
    def feature_importances_(self):
        return np.array([])

    def fit(self, *args, **kwargs):
        raise NotImplementedError("Training không khả dụng trên Android")


# ── Fake module-level API (lgb.LGBMClassifier) ───────────────────────────────

# Một số code dùng lgb.train() hoặc lgb.Dataset() — stub để tránh crash
class Dataset:
    def __init__(self, *args, **kwargs): pass

def train(*args, **kwargs):
    raise NotImplementedError("lgb.train() không khả dụng trên Android")

def cv(*args, **kwargs):
    raise NotImplementedError("lgb.cv() không khả dụng trên Android")