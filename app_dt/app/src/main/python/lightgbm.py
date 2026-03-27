class LGBMClassifier:
    """Stub class - ONNX inference is used instead"""
    def __init__(self, **kwargs):
        pass
    
    def fit(self, X, y):
        raise RuntimeError("LightGBM not available on mobile - use ONNX instead")
    
    def predict(self, X):
        raise RuntimeError("LightGBM not available on mobile - use ONNX instead")
    
    def predict_proba(self, X):
        raise RuntimeError("LightGBM not available on mobile - use ONNX instead")


class LGBMRegressor:
    """Stub class - ONNX inference is used instead"""
    def __init__(self, **kwargs):
        pass
    
    def fit(self, X, y):
        raise RuntimeError("LightGBM not available on mobile - use ONNX instead")
    
    def predict(self, X):
        raise RuntimeError("LightGBM not available on mobile - use ONNX instead")


class LGBMRanker:
    """Stub class - ONNX inference is used instead"""
    def __init__(self, **kwargs):
        pass


class LGBMModel:
    """Stub class - ONNX inference is used instead"""
    def __init__(self, **kwargs):
        pass


class Dataset:
    """Stub class - ONNX inference is used instead"""
    def __init__(self, data, label=None):
        pass


def train(params, train_set, num_boost_round=100, **kwargs):
    """Stub function - ONNX inference is used instead"""
    raise RuntimeError("LightGBM not available on mobile - use ONNX instead")


def cv(params, train_set, num_boost_round=100, nfold=5, **kwargs):
    """Stub function - ONNX inference is used instead"""
    raise RuntimeError("LightGBM not available on mobile - use ONNX instead")