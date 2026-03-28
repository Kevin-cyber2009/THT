class LGBMClassifier:
    def __init__(self, **kwargs):
        pass
    
    def fit(self, X, y):
        raise RuntimeError("LightGBM not available on mobile - use ONNX instead")
    
    def predict(self, X):
        raise RuntimeError("LightGBM not available on mobile - use ONNX instead")
    
    def predict_proba(self, X):
        raise RuntimeError("LightGBM not available on mobile - use ONNX instead")


class LGBMRegressor:
    def __init__(self, **kwargs):
        pass
    
    def fit(self, X, y):
        raise RuntimeError("LightGBM not available on mobile - use ONNX instead")
    
    def predict(self, X):
        raise RuntimeError("LightGBM not available on mobile - use ONNX instead")


class LGBMRanker:
    def __init__(self, **kwargs):
        pass


class LGBMModel:
    def __init__(self, **kwargs):
        pass


class Dataset:
    def __init__(self, data, label=None):
        pass


def train(params, train_set, num_boost_round=100, **kwargs):
    raise RuntimeError("LightGBM not available on mobile - use ONNX instead")


def cv(params, train_set, num_boost_round=100, nfold=5, **kwargs):
    raise RuntimeError("LightGBM not available on mobile - use ONNX instead")