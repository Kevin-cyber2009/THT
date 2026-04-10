import logging
import os
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

try:
    from .utils import load_config
except ImportError:
    from utils import load_config


logger = logging.getLogger('hybrid_detector.deep_features')


def _resolve_model_path(config: dict, model_filename: str) -> Optional[str]:
    candidates = []

    deep_config = config.get('deep_learning', {})
    injected = deep_config.get('models_dir')
    if injected:
        candidates.append(os.path.join(injected, model_filename))

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates.append(os.path.join(base, 'models', model_filename))
    candidates.append(os.path.join(base, 'assets', 'models', model_filename))

    for path in candidates:
        if os.path.exists(path):
            logger.info("Found ONNX deep model at: %s", path)
            return path

    logger.warning("Model %s not found in candidates: %s", model_filename, candidates)
    return None


class DeepFeatureExtractor:
    def __init__(self, config: Optional[dict] = None, model_path: Optional[str] = None):
        if config is None:
            config = load_config()

        self.deep_config = config.get('deep_learning', {})
        self.model_type = self.deep_config.get('model_type', 'resnet50')

        try:
            import onnxruntime as rt

            if model_path is None:
                model_path = _resolve_model_path(config, f"{self.model_type}_features.onnx")

            if model_path is None or not os.path.exists(model_path):
                raise FileNotFoundError(f"ONNX model not found for {self.model_type}")

            self.session = rt.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
        except Exception as exc:
            logger.error("Error loading ONNX model: %s", exc)
            raise

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        frame_uint8 = (frame * 255).astype(np.uint8)
        img = Image.fromarray(frame_uint8)
        img = img.resize((256, 256), Image.BILINEAR)
        left = (256 - 224) // 2
        top = (256 - 224) // 2
        img = img.crop((left, top, left + 224, top + 224))

        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = (img_array - self.mean) / self.std
        img_array = img_array.transpose(2, 0, 1)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array.astype(np.float32)

    def extract_frame_features(self, frame: np.ndarray) -> np.ndarray:
        input_tensor = self._preprocess_frame(frame)
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        return outputs[0].flatten()

    def extract_video_features(self, frames: np.ndarray, sample_frames: int = 10) -> Dict[str, float]:
        num_frames = len(frames)
        indices = np.linspace(0, num_frames - 1, sample_frames).astype(int) if num_frames > sample_frames else np.arange(num_frames)

        frame_features = []
        for idx in indices:
            try:
                frame_features.append(self.extract_frame_features(frames[idx]))
            except Exception as exc:
                logger.warning("Error extracting features from frame %s: %s", idx, exc)

        if not frame_features:
            return self._get_zero_features()

        features_matrix = np.array(frame_features)
        features_dict: Dict[str, float] = {
            'deep_feat_mean': float(np.mean(features_matrix)),
            'deep_feat_std': float(np.std(features_matrix)),
            'deep_feat_max': float(np.max(features_matrix)),
            'deep_feat_min': float(np.min(features_matrix)),
        }

        temporal_var = np.var(features_matrix, axis=0)
        features_dict['deep_temporal_var_mean'] = float(np.mean(temporal_var))
        features_dict['deep_temporal_var_std'] = float(np.std(temporal_var))

        l2_norms = np.linalg.norm(features_matrix, axis=1)
        features_dict['deep_l2_norm_mean'] = float(np.mean(l2_norms))
        features_dict['deep_l2_norm_std'] = float(np.std(l2_norms))

        if len(frame_features) > 1:
            similarities = []
            for i in range(len(frame_features) - 1):
                sim = np.dot(frame_features[i], frame_features[i + 1]) / (
                    np.linalg.norm(frame_features[i]) * np.linalg.norm(frame_features[i + 1]) + 1e-10
                )
                similarities.append(sim)
            features_dict['deep_similarity_mean'] = float(np.mean(similarities))
            features_dict['deep_similarity_std'] = float(np.std(similarities))
        else:
            features_dict['deep_similarity_mean'] = 0.0
            features_dict['deep_similarity_std'] = 0.0

        features_dict['deep_sparsity'] = float(np.mean(np.abs(features_matrix) < 0.01))
        return features_dict

    def _get_zero_features(self) -> Dict[str, float]:
        return {
            'deep_feat_mean': 0.0,
            'deep_feat_std': 0.0,
            'deep_feat_max': 0.0,
            'deep_feat_min': 0.0,
            'deep_temporal_var_mean': 0.0,
            'deep_temporal_var_std': 0.0,
            'deep_l2_norm_mean': 0.0,
            'deep_l2_norm_std': 0.0,
            'deep_similarity_mean': 0.0,
            'deep_similarity_std': 0.0,
            'deep_sparsity': 0.0,
        }

    def get_feature_names(self) -> list:
        return [
            'deep_feat_mean',
            'deep_feat_std',
            'deep_feat_max',
            'deep_feat_min',
            'deep_temporal_var_mean',
            'deep_temporal_var_std',
            'deep_l2_norm_mean',
            'deep_l2_norm_std',
            'deep_similarity_mean',
            'deep_similarity_std',
            'deep_sparsity',
        ]


class EnsembleDeepExtractor:
    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = load_config()

        self.models = []
        model_types = config.get('deep_learning', {}).get('ensemble_models', ['resnet50'])

        for model_type in model_types:
            try:
                import copy

                cfg_copy = copy.deepcopy(config)
                cfg_copy['deep_learning']['model_type'] = model_type
                model_path = _resolve_model_path(cfg_copy, f"{model_type}_features.onnx")
                if model_path is None:
                    continue
                self.models.append(DeepFeatureExtractor(cfg_copy, model_path))
            except Exception as exc:
                logger.warning("Failed to load %s: %s", model_type, exc)

        if not self.models:
            raise RuntimeError("No ONNX models could be loaded for ensemble")

    def extract_video_features(self, frames: np.ndarray, sample_frames: int = 10) -> Dict[str, float]:
        all_features: Dict[str, float] = {}

        for extractor in self.models:
            features = extractor.extract_video_features(frames, sample_frames)
            for key, value in features.items():
                all_features[f"{extractor.model_type}_{key}"] = value

        canonical_names = self.models[0].get_feature_names()
        for base_key in canonical_names:
            values = []
            for extractor in self.models:
                full_key = f"{extractor.model_type}_{base_key}"
                if full_key in all_features:
                    values.append(all_features[full_key])
            all_features[base_key] = float(np.mean(values)) if values else 0.0

        return all_features
