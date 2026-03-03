import os, sys

# Thêm đường dẫn src vào Python path
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from src.utils import load_config
from src.classifier import VideoClassifier
from src.features import FeatureExtractor
from src.fusion import ScoreFusion

_classifier = None
_extractor  = None
_fusion     = None

def load_models(model_path: str, config_path: str) -> str:
    global _classifier, _extractor, _fusion
    try:
        config = load_config(config_path)
        config.setdefault('preprocessing', {}).update({
            'max_frames': 30, 'target_fps': 3,
            'resize_width': 320, 'resize_height': 180
        })
        _classifier = VideoClassifier(config)
        _classifier.load(model_path)
        _extractor  = FeatureExtractor(config)
        _fusion     = ScoreFusion(config)
        return 'OK'
    except Exception as e:
        return f'ERROR: {e}'

def analyze_video(video_path: str) -> dict:
    if not _classifier:
        return {'error': 'Model chua duoc load'}
    try:
        features, metadata = _extractor.extract_from_video(video_path)
        names   = _classifier.feature_names or _extractor.get_feature_names()
        vector  = _extractor.features_to_vector(features, names)
        pred, prob = _classifier.predict(vector.reshape(1, -1))

        artifact = _fusion.compute_artifact_score(features)
        reality  = _fusion.compute_reality_score(features)
        fusion   = _fusion.fuse_scores(artifact, reality, 0.5)
        explain  = _fusion.generate_explanation(features, fusion)

        return {
            'prediction':       'FAKE' if pred[0] == 1 else 'REAL',
            'probability_fake':  float(prob[0]),
            'confidence':        fusion['confidence'],
            'artifact_score':    float(artifact),
            'reality_score':     float(reality),
            'explanations':      explain[:5],
        }
    except Exception as e:
        return {'error': str(e)}