import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .forensic import ForensicAnalyzer
from .preprocessing import VideoPreprocessor
from .reality_engine import RealityEngine
from .utils import load_config, normalize_array


logger = logging.getLogger('hybrid_detector.features')


class FeatureExtractor:
    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = load_config()

        self.config = config
        self.feature_config = config.get('features', {})
        self.normalization = self.feature_config.get('normalization', 'standard')

        logger.info("Initializing traditional feature extractors...")
        self.preprocessor = VideoPreprocessor(config)
        self.forensic = ForensicAnalyzer(config)
        self.reality = RealityEngine(config)

        self.deep_extractor = None
        self.use_deep = config.get('features', {}).get('use_deep_features', False)

        if self.use_deep:
            deep_loaded = False
            prefer_onnx = config.get('deep_learning', {}).get('prefer_onnx', False)
            module_order = ['onnx', 'torch'] if prefer_onnx else ['torch', 'onnx']

            for backend in module_order:
                try:
                    if backend == 'onnx':
                        logger.info("Attempting to load ONNX deep learning module...")
                        from .deep_features_onnx import DeepFeatureExtractor, EnsembleDeepExtractor
                    else:
                        logger.info("Attempting to load PyTorch deep learning module...")
                        from .deep_features import DeepFeatureExtractor, EnsembleDeepExtractor

                    use_ensemble = config.get('deep_learning', {}).get('use_ensemble', False)
                    if use_ensemble:
                        logger.info("Initializing Ensemble Deep Learning extractors (%s)...", backend.upper())
                        self.deep_extractor = EnsembleDeepExtractor(config)
                    else:
                        logger.info("Initializing Deep Learning extractor (%s)...", backend.upper())
                        self.deep_extractor = DeepFeatureExtractor(config)

                    logger.info("Deep learning ENABLED (%s)", backend.upper())
                    deep_loaded = True
                    break
                except ImportError as exc:
                    logger.warning("%s backend not available: %s", backend.upper(), exc)
                except Exception as exc:
                    logger.error("Error initializing %s deep learning: %s", backend.upper(), exc)

            if not deep_loaded:
                logger.warning("Deep learning features DISABLED - using traditional only")
                self.use_deep = False
                self.deep_extractor = None
        else:
            logger.info("Deep learning features disabled in config")

        self.expected_dim = config.get('features', {}).get('expected_dimension', 39) if self.use_deep else 28

        logger.info("FeatureExtractor initialized")
        logger.info("  Mode: %s", 'HYBRID (Traditional + Deep)' if self.use_deep else 'TRADITIONAL ONLY')
        logger.info("  Expected dimension: %s", self.expected_dim)

    def extract_features(self, frames: np.ndarray) -> Dict[str, float]:
        all_features: Dict[str, float] = {}

        logger.debug("Extracting forensic features...")
        all_features.update(self.forensic.analyze(frames))

        logger.debug("Extracting reality features...")
        all_features.update(self.reality.analyze(frames))

        logger.info("Extracted %s traditional features", len(all_features))

        if self.use_deep and self.deep_extractor is not None:
            try:
                logger.debug("Extracting deep learning features...")
                sample_frames = self.config.get('deep_learning', {}).get('sample_frames', 10)
                deep_feats = self.deep_extractor.extract_video_features(frames, sample_frames)
                all_features.update(deep_feats)
                logger.info("Extracted %s deep learning features", len(deep_feats))
            except Exception as exc:
                logger.error("Error extracting deep features: %s", exc)
                logger.warning("Continuing with traditional features only")

        logger.info("Total features extracted: %s", len(all_features))
        return all_features

    def extract_from_video(self, video_path: str) -> Tuple[Dict[str, float], dict]:
        logger.info("Extracting features from video: %s", video_path)

        frames, metadata = self.preprocessor.preprocess(video_path)
        if len(frames) < 10:
            logger.warning("Video short (%s frames), padding...", len(frames))
            frames = self.preprocessor.handle_short_video(frames, min_frames=10)

        features = self.extract_features(frames)
        return features, metadata

    def features_to_vector(self, features: Dict[str, float], feature_names: Optional[List[str]] = None) -> np.ndarray:
        if feature_names is None:
            feature_names = sorted(features.keys())

        vector = []
        for name in feature_names:
            value = features.get(name, 0.0)
            if np.isnan(value) or np.isinf(value):
                value = 0.0
            vector.append(value)

        return np.array(vector, dtype=np.float32)

    def normalize_features(self, feature_vector: np.ndarray) -> np.ndarray:
        return normalize_array(feature_vector, method=self.normalization)

    def get_feature_names(self) -> List[str]:
        forensic_names = [
            'fft_mean', 'fft_std', 'fft_max', 'fft_high_freq_energy', 'fft_radial_slope',
            'dct_mean', 'dct_std', 'dct_dc_mean', 'dct_ac_energy',
            'prnu_mean', 'prnu_std', 'prnu_autocorr', 'prnu_temporal_consistency',
            'flow_mean_magnitude', 'flow_std_magnitude', 'flow_smoothness', 'flow_temporal_consistency',
        ]

        reality_names = [
            'entropy_mean', 'entropy_std', 'entropy_slope',
            'fractal_dim_mean', 'fractal_dim_std',
            'causal_prediction_error', 'causal_predictability',
            'compression_mean', 'compression_std', 'compression_delta_mean', 'complexity_mean',
        ]

        traditional_names = forensic_names + reality_names
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
        return traditional_names

    def get_feature_info(self) -> Dict[str, Any]:
        feature_names = self.get_feature_names()
        info: Dict[str, Any] = {
            'total_features': len(feature_names),
            'traditional_features': 28,
            'deep_features': len(feature_names) - 28 if self.use_deep else 0,
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
