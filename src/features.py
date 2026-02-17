# src/features.py
"""
Module features: Aggregator tổng hợp forensic + reality + deep learning features
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
from typing import Tuple
from .preprocessing import VideoPreprocessor
from .forensic import ForensicAnalyzer
from .reality_engine import RealityEngine
from .utils import load_config, normalize_array


logger = logging.getLogger('hybrid_detector.features')


class FeatureExtractor:
    """
    Class tổng hợp trích xuất ALL features từ video
    - Traditional: Forensic + Reality (35 features)
    - Deep Learning: CNN features (11 features)
    Total: 46 features (có thể extend với ensemble)
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Khởi tạo FeatureExtractor
        
        Args:
            config: Dictionary cấu hình
        """
        if config is None:
            config = load_config()
        
        self.config = config
        self.feature_config = config.get('features', {})
        self.normalization = self.feature_config.get('normalization', 'standard')
        
        # Initialize traditional analyzers
        logger.info("Initializing traditional feature extractors...")
        self.preprocessor = VideoPreprocessor(config)
        self.forensic = ForensicAnalyzer(config)
        self.reality = RealityEngine(config)
        
        # Initialize deep learning extractor (optional)
        self.deep_extractor = None
        self.use_deep = config.get('features', {}).get('use_deep_features', False)
        
        if self.use_deep:
            try:
                logger.info("Attempting to load deep learning module...")
                from .deep_features import DeepFeatureExtractor, EnsembleDeepExtractor
                
                use_ensemble = config.get('deep_learning', {}).get('use_ensemble', False)
                
                if use_ensemble:
                    logger.info("Initializing Ensemble Deep Learning extractors...")
                    self.deep_extractor = EnsembleDeepExtractor(config)
                else:
                    logger.info("Initializing Deep Learning extractor...")
                    self.deep_extractor = DeepFeatureExtractor(config)
                
                logger.info("✓ Deep learning ENABLED")
                
            except ImportError as e:
                logger.warning(f"Cannot import deep_features: {e}")
                logger.warning("Deep learning features DISABLED - using traditional only")
                self.use_deep = False
                self.deep_extractor = None
            except Exception as e:
                logger.error(f"Error initializing deep learning: {e}")
                logger.warning("Deep learning features DISABLED - using traditional only")
                self.use_deep = False
                self.deep_extractor = None
        else:
            logger.info("Deep learning features disabled in config")
        
        # Set expected dimension
        if self.use_deep:
            self.expected_dim = config.get('features', {}).get('expected_dimension', 46)
        else:
            self.expected_dim = 35  # Traditional only
        
        logger.info(f"✓ FeatureExtractor initialized")
        logger.info(f"  Mode: {'HYBRID (Traditional + Deep)' if self.use_deep else 'TRADITIONAL ONLY'}")
        logger.info(f"  Expected dimension: {self.expected_dim}")
    
    def extract_features(self, frames: np.ndarray) -> Dict[str, float]:
        """
        Trích xuất TẤT CẢ features từ frames
        
        Args:
            frames: Array frames normalized
            
        Returns:
            Dictionary chứa tất cả features (traditional + deep nếu enabled)
        """
        all_features = {}
        
        # 1. Extract Traditional Features
        logger.debug("Extracting forensic features...")
        forensic_feats = self.forensic.analyze(frames)
        all_features.update(forensic_feats)
        
        logger.debug("Extracting reality features...")
        reality_feats = self.reality.analyze(frames)
        all_features.update(reality_feats)
        
        traditional_count = len(all_features)
        logger.info(f"✓ Extracted {traditional_count} traditional features")
        
        # 2. Extract Deep Learning Features (if enabled)
        if self.use_deep and self.deep_extractor is not None:
            try:
                logger.debug("Extracting deep learning features...")
                sample_frames = self.config.get('deep_learning', {}).get('sample_frames', 10)
                deep_feats = self.deep_extractor.extract_video_features(frames, sample_frames)
                all_features.update(deep_feats)
                
                deep_count = len(deep_feats)
                logger.info(f"✓ Extracted {deep_count} deep learning features")
                
            except Exception as e:
                logger.error(f"Error extracting deep features: {e}")
                logger.warning("Continuing with traditional features only")
        
        total_count = len(all_features)
        logger.info(f"✓ Total features extracted: {total_count}")
        
        return all_features
    
    def extract_from_video(self, video_path: str) -> Tuple[Dict[str, float], dict]:
        """
        Pipeline đầy đủ: video -> frames -> features (traditional + deep)
        
        Args:
            video_path: Đường dẫn video
            
        Returns:
            Tuple (features_dict, metadata)
        """
        logger.info(f"Extracting features from video: {video_path}")
        
        # Preprocess
        frames, metadata = self.preprocessor.preprocess(video_path)
        
        # Handle short videos
        if len(frames) < 10:
            logger.warning(f"Video ngắn ({len(frames)} frames), padding...")
            frames = self.preprocessor.handle_short_video(frames, min_frames=10)
        
        # Extract features (traditional + deep)
        features = self.extract_features(frames)
        
        return features, metadata
    
    def features_to_vector(
        self, 
        features: Dict[str, float],
        feature_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Convert features dict thành numpy vector theo thứ tự cố định
        
        Args:
            features: Dictionary features
            feature_names: List tên features theo thứ tự (nếu None, tự động sort)
            
        Returns:
            1D numpy array
        """
        if feature_names is None:
            # Sort keys để đảm bảo thứ tự nhất quán
            feature_names = sorted(features.keys())
        
        vector = []
        for name in feature_names:
            value = features.get(name, 0.0)
            # Handle NaN/Inf
            if np.isnan(value) or np.isinf(value):
                value = 0.0
            vector.append(value)
        
        return np.array(vector, dtype=np.float32)
    
    def normalize_features(self, feature_vector: np.ndarray) -> np.ndarray:
        """
        Chuẩn hóa feature vector
        
        Args:
            feature_vector: 1D array features
            
        Returns:
            Normalized array
        """
        return normalize_array(feature_vector, method=self.normalization)
    
    def get_feature_names(self) -> List[str]:
        """
        Lấy danh sách tên features theo thứ tự chuẩn
        
        Returns:
            List tên features (traditional + deep nếu enabled)
        """
        # Traditional features
        forensic_names = [
            'fft_mean', 'fft_std', 'fft_max', 'fft_high_freq_energy', 'fft_radial_slope',
            'dct_mean', 'dct_std', 'dct_dc_mean', 'dct_ac_energy',
            'prnu_mean', 'prnu_std', 'prnu_autocorr', 'prnu_temporal_consistency',
            'flow_mean_magnitude', 'flow_std_magnitude', 'flow_smoothness', 'flow_temporal_consistency'
        ]
        
        reality_names = [
            'entropy_mean', 'entropy_std', 'entropy_slope',
            'fractal_dim_mean', 'fractal_dim_std',
            'causal_prediction_error', 'causal_predictability',
            'compression_mean', 'compression_std', 'compression_delta_mean', 'complexity_mean'
        ]
        
        traditional_names = forensic_names + reality_names
        
        # Deep learning features (if enabled)
        if self.use_deep and self.deep_extractor is not None:
            deep_names = [
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
            return traditional_names + deep_names
        else:
            return traditional_names
    
    def get_feature_info(self) -> Dict[str, Any]:
        """
        Lấy thông tin chi tiết về features
        
        Returns:
            Dictionary chứa info về features
        """
        feature_names = self.get_feature_names()
        
        info = {
            'total_features': len(feature_names),
            'traditional_features': 35,
            'deep_features': len(feature_names) - 35 if self.use_deep else 0,
            'use_deep_learning': self.use_deep,
            'feature_names': feature_names,
            'expected_dimension': self.expected_dim,
        }
        
        if self.use_deep and self.deep_extractor is not None:
            if hasattr(self.deep_extractor, 'model_type'):
                info['deep_model'] = self.deep_extractor.model_type
            elif hasattr(self.deep_extractor, 'models'):
                info['deep_model'] = 'ensemble'
                info['ensemble_models'] = [m.model_type for m in self.deep_extractor.models]
        
        return info


