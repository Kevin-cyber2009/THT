import numpy as np
from PIL import Image
import logging
import os
from typing import Dict, Any, Optional

try:
    from .utils import load_config
except ImportError:
    from utils import load_config


logger = logging.getLogger('hybrid_detector.deep_features')


class DeepFeatureExtractor:
    
    def __init__(self, config: Optional[dict] = None, model_path: Optional[str] = None):
        if config is None:
            config = load_config()
        
        self.deep_config = config.get('deep_learning', {})
        self.model_type = self.deep_config.get('model_type', 'resnet50')
        
        # Initialize ONNX Runtime session
        try:
            import onnxruntime as rt
            
            # Determine model path
            if model_path is None:
                # Default model paths in Android assets
                base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                assets_models = os.path.join(base_path, 'assets', 'models')
                
                if self.model_type == 'resnet50':
                    model_path = os.path.join(assets_models, 'resnet50_features.onnx')
                elif self.model_type == 'efficientnet_b0':
                    model_path = os.path.join(assets_models, 'efficientnet_b0_features.onnx')
                else:
                    model_path = os.path.join(assets_models, 'resnet50_features.onnx')
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"ONNX model not found: {model_path}")
            
            # Create ONNX Runtime session
            self.session = rt.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            logger.info(f"DeepFeatureExtractor initialized: {self.model_type} (ONNX)")
            logger.info(f"  Model path: {model_path}")
            logger.info(f"  Input name: {self.input_name}")
            logger.info(f"  Output name: {self.output_name}")
            
        except ImportError:
            logger.error("ONNX Runtime not available. Cannot extract deep features.")
            raise
        except Exception as e:
            logger.error(f"Error loading ONNX model: {e}")
            raise
        
        # ImageNet normalization constants
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        frame_uint8 = (frame * 255).astype(np.uint8)
        img = Image.fromarray(frame_uint8)
        
        # Resize and crop
        img = img.resize((256, 256), Image.BILINEAR)
        
        # Center crop to 224x224
        left = (256 - 224) // 2
        top = (256 - 224) // 2
        img = img.crop((left, top, left + 224, top + 224))
        
        # Convert to numpy and normalize
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        img_array = (img_array - self.mean) / self.std
        
        # HWC -> CHW
        img_array = img_array.transpose(2, 0, 1)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array.astype(np.float32)
    
    def extract_frame_features(self, frame: np.ndarray) -> np.ndarray:
        input_tensor = self._preprocess_frame(frame)
        
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        
        features = outputs[0].flatten()
        
        return features
    
    def extract_video_features(
        self,
        frames: np.ndarray,
        sample_frames: int = 10
    ) -> Dict[str, float]:
        num_frames = len(frames)
        
        if num_frames > sample_frames:
            indices = np.linspace(0, num_frames - 1, sample_frames).astype(int)
        else:
            indices = np.arange(num_frames)
        
        frame_features = []
        
        logger.debug(f"Extracting deep features from {len(indices)} frames using ONNX...")
        
        for idx in indices:
            try:
                feat = self.extract_frame_features(frames[idx])
                frame_features.append(feat)
            except Exception as e:
                logger.warning(f"Error extracting features from frame {idx}: {e}")
                continue
        
        if len(frame_features) == 0:
            logger.warning("No features extracted, returning zeros")
            return self._get_zero_features()
        
        features_matrix = np.array(frame_features)
        
        features_dict = {}
        
        features_dict['deep_feat_mean'] = float(np.mean(features_matrix))
        features_dict['deep_feat_std'] = float(np.std(features_matrix))
        features_dict['deep_feat_max'] = float(np.max(features_matrix))
        features_dict['deep_feat_min'] = float(np.min(features_matrix))
        
        temporal_var = np.var(features_matrix, axis=0)
        features_dict['deep_temporal_var_mean'] = float(np.mean(temporal_var))
        features_dict['deep_temporal_var_std'] = float(np.std(temporal_var))
        
        l2_norms = np.linalg.norm(features_matrix, axis=1)
        features_dict['deep_l2_norm_mean'] = float(np.mean(l2_norms))
        features_dict['deep_l2_norm_std'] = float(np.std(l2_norms))
        
        if len(frame_features) > 1:
            similarities = []
            for i in range(len(frame_features) - 1):
                sim = np.dot(frame_features[i], frame_features[i+1]) / (
                    np.linalg.norm(frame_features[i]) * np.linalg.norm(frame_features[i+1]) + 1e-10
                )
                similarities.append(sim)
            
            features_dict['deep_similarity_mean'] = float(np.mean(similarities))
            features_dict['deep_similarity_std'] = float(np.std(similarities))
        else:
            features_dict['deep_similarity_mean'] = 0.0
            features_dict['deep_similarity_std'] = 0.0
        
        threshold = 0.01
        sparsity = np.mean(np.abs(features_matrix) < threshold)
        features_dict['deep_sparsity'] = float(sparsity)
        
        logger.debug(f"Deep features extracted: {len(features_dict)} features")
        
        return features_dict
    
    def _get_zero_features(self) -> Dict[str, float]:
        """Return zero features when extraction fails"""
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
        """Return list of feature names"""
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
        
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        assets_models = os.path.join(base_path, 'assets', 'models')
        
        for model_type in model_types:
            logger.info(f"Loading {model_type} (ONNX)...")
            
            if model_type == 'resnet50':
                model_path = os.path.join(assets_models, 'resnet50_features.onnx')
            elif model_type == 'efficientnet_b0':
                model_path = os.path.join(assets_models, 'efficientnet_b0_features.onnx')
            else:
                logger.warning(f"Unknown model type: {model_type}, skipping")
                continue
            
            if not os.path.exists(model_path):
                logger.warning(f"ONNX model not found: {model_path}, skipping")
                continue
            
            try:
                extractor = DeepFeatureExtractor(config, model_path)
                self.models.append(extractor)
            except Exception as e:
                logger.warning(f"Failed to load {model_type}: {e}")
                continue
        
        if len(self.models) == 0:
            raise RuntimeError("No ONNX models could be loaded for ensemble")
        
        logger.info(f"EnsembleDeepExtractor initialized with {len(self.models)} models")
    
    def extract_video_features(
        self,
        frames: np.ndarray,
        sample_frames: int = 10
    ) -> Dict[str, float]:
        all_features = {}
        
        for i, extractor in enumerate(self.models):
            logger.info(f"Extracting features from model {i+1}/{len(self.models)}...")
            features = extractor.extract_video_features(frames, sample_frames)
            
            model_name = extractor.model_type
            for key, value in features.items():
                new_key = f"{model_name}_{key}"
                all_features[new_key] = value
        
        if len(self.models) > 1:
            base_keys = self.models[0].get_feature_names()
            
            for base_key in base_keys:
                values = []
                for extractor in self.models:
                    model_name = extractor.model_type
                    full_key = f"{model_name}_{base_key}"
                    if full_key in all_features:
                        values.append(all_features[full_key])
                
                if len(values) > 0:
                    all_features[f'ensemble_{base_key}_mean'] = float(np.mean(values))
                    all_features[f'ensemble_{base_key}_std'] = float(np.std(values))
        
        logger.info(f"Ensemble extraction complete: {len(all_features)} features")
        
        return all_features