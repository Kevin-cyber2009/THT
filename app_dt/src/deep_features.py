#!/usr/bin/env python3
# src/deep_features.py
"""
Module deep_features: Extract features từ pretrained CNN models
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import logging
from typing import Dict, Any, Optional

try:
    from .utils import load_config
except ImportError:
    from utils import load_config


logger = logging.getLogger('hybrid_detector.deep_features')


class DeepFeatureExtractor:
    """
    Class extract deep features từ pretrained CNNs
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Khởi tạo DeepFeatureExtractor
        
        Args:
            config: Dictionary cấu hình
        """
        if config is None:
            config = load_config()
        
        self.deep_config = config.get('deep_learning', {})
        self.model_type = self.deep_config.get('model_type', 'resnet50')
        self.use_gpu = self.deep_config.get('use_gpu', True) and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load pretrained model
        self.model = self._load_model()
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"DeepFeatureExtractor initialized: {self.model_type}, device: {self.device}")
    
    def _load_model(self):
        """Load pretrained model và remove classifier"""
        
        if self.model_type == 'resnet50':
            model = models.resnet50(weights='IMAGENET1K_V1')
            # Remove final FC layer
            model = nn.Sequential(*list(model.children())[:-1])
            
        elif self.model_type == 'resnet18':
            model = models.resnet18(weights='IMAGENET1K_V1')
            model = nn.Sequential(*list(model.children())[:-1])
            
        elif self.model_type == 'efficientnet_b0':
            model = models.efficientnet_b0(weights='IMAGENET1K_V1')
            # Remove classifier
            model.classifier = nn.Identity()
            
        elif self.model_type == 'efficientnet_b3':
            model = models.efficientnet_b3(weights='IMAGENET1K_V1')
            model.classifier = nn.Identity()
            
        elif self.model_type == 'vgg16':
            model = models.vgg16(weights='IMAGENET1K_V1')
            # Remove classifier, keep only features
            model = model.features
            
        else:
            logger.warning(f"Unknown model type: {self.model_type}, using resnet50")
            model = models.resnet50(weights='IMAGENET1K_V1')
            model = nn.Sequential(*list(model.children())[:-1])
        
        return model
    
    def extract_frame_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract features từ 1 frame
        
        Args:
            frame: Frame numpy array (H, W, C), normalized [0, 1]
            
        Returns:
            Feature vector
        """
        # Convert to PIL Image
        frame_uint8 = (frame * 255).astype(np.uint8)
        img = Image.fromarray(frame_uint8)
        
        # Transform
        img_tensor = self.transform(img).unsqueeze(0)  # Add batch dimension
        img_tensor = img_tensor.to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(img_tensor)
        
        # Flatten
        features = features.cpu().numpy().flatten()
        
        return features
    
    def extract_video_features(
        self,
        frames: np.ndarray,
        sample_frames: int = 10
    ) -> Dict[str, float]:
        """
        Extract deep features từ video frames
        
        Args:
            frames: Array frames (N, H, W, C), normalized [0, 1]
            sample_frames: Số frames để sample
            
        Returns:
            Dictionary chứa deep learning features
        """
        num_frames = len(frames)
        
        # Sample frames uniformly
        if num_frames > sample_frames:
            indices = np.linspace(0, num_frames - 1, sample_frames).astype(int)
        else:
            indices = np.arange(num_frames)
        
        # Extract features từ sampled frames
        frame_features = []
        
        logger.debug(f"Extracting deep features from {len(indices)} frames...")
        
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
        
        # Stack features
        features_matrix = np.array(frame_features)  # (num_frames, feature_dim)
        
        # Aggregate statistics
        features_dict = {}
        
        # Basic statistics
        features_dict['deep_feat_mean'] = float(np.mean(features_matrix))
        features_dict['deep_feat_std'] = float(np.std(features_matrix))
        features_dict['deep_feat_max'] = float(np.max(features_matrix))
        features_dict['deep_feat_min'] = float(np.min(features_matrix))
        
        # Temporal statistics (variance across frames)
        temporal_var = np.var(features_matrix, axis=0)  # Variance per feature dimension
        features_dict['deep_temporal_var_mean'] = float(np.mean(temporal_var))
        features_dict['deep_temporal_var_std'] = float(np.std(temporal_var))
        
        # L2 norm statistics
        l2_norms = np.linalg.norm(features_matrix, axis=1)
        features_dict['deep_l2_norm_mean'] = float(np.mean(l2_norms))
        features_dict['deep_l2_norm_std'] = float(np.std(l2_norms))
        
        # Pairwise similarity (consecutive frames)
        if len(frame_features) > 1:
            similarities = []
            for i in range(len(frame_features) - 1):
                # Cosine similarity
                sim = np.dot(frame_features[i], frame_features[i+1]) / (
                    np.linalg.norm(frame_features[i]) * np.linalg.norm(frame_features[i+1]) + 1e-10
                )
                similarities.append(sim)
            
            features_dict['deep_similarity_mean'] = float(np.mean(similarities))
            features_dict['deep_similarity_std'] = float(np.std(similarities))
        else:
            features_dict['deep_similarity_mean'] = 0.0
            features_dict['deep_similarity_std'] = 0.0
        
        # Sparsity (percentage of near-zero values)
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
    """
    Extract features từ multiple CNN models và ensemble
    """
    
    def __init__(self, config: Optional[dict] = None):
        """Initialize ensemble of models"""
        if config is None:
            config = load_config()
        
        self.models = []
        model_types = config.get('deep_learning', {}).get('ensemble_models', ['resnet50'])
        
        for model_type in model_types:
            logger.info(f"Loading {model_type}...")
            config_copy = config.copy()
            config_copy['deep_learning']['model_type'] = model_type
            extractor = DeepFeatureExtractor(config_copy)
            self.models.append(extractor)
        
        logger.info(f"EnsembleDeepExtractor initialized with {len(self.models)} models")
    
    def extract_video_features(
        self,
        frames: np.ndarray,
        sample_frames: int = 10
    ) -> Dict[str, float]:
        """Extract features từ all models và combine"""
        
        all_features = {}
        
        for i, extractor in enumerate(self.models):
            logger.info(f"Extracting features from model {i+1}/{len(self.models)}...")
            features = extractor.extract_video_features(frames, sample_frames)
            
            # Rename features với prefix
            model_name = extractor.model_type
            for key, value in features.items():
                new_key = f"{model_name}_{key}"
                all_features[new_key] = value
        
        # Add ensemble statistics (average across models)
        if len(self.models) > 1:
            # Get common feature keys
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


# Test when run directly
if __name__ == '__main__':
    print("Testing DeepFeatureExtractor...")
    
    # Test import
    try:
        extractor = DeepFeatureExtractor()
        print(f"✓ DeepFeatureExtractor initialized: {extractor.model_type}")
        print(f"✓ Device: {extractor.device}")
        print(f"✓ Feature names: {extractor.get_feature_names()}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()