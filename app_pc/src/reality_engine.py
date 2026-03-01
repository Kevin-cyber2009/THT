# src/reality_engine.py
"""
Module reality_engine: Kiểm tra tuân thủ thực tại vật lý
(Entropy, Fractal, Causal Motion, Information Conservation)
"""

import cv2
import numpy as np
from scipy import ndimage
from sklearn.linear_model import Ridge
from typing import Dict, Any, Optional
import logging

from .utils import load_config, safe_divide


logger = logging.getLogger('hybrid_detector.reality_engine')


class RealityEngine:
    """
    Class kiểm tra reality compliance: entropy, fractal, causality, conservation
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Khởi tạo RealityEngine
        
        Args:
            config: Dictionary cấu hình
        """
        if config is None:
            config = load_config()
        
        self.reality_config = config.get('reality_engine', {})
        
        # Entropy config
        self.entropy_scales = self.reality_config.get('entropy_scales', 4)
        self.entropy_window = self.reality_config.get('entropy_window_size', 16)
        
        # Fractal config
        self.fractal_boxes = self.reality_config.get('fractal_box_sizes', [2, 4, 8, 16, 32])
        self.edge_low = self.reality_config.get('fractal_edge_threshold_low', 50)
        self.edge_high = self.reality_config.get('fractal_edge_threshold_high', 150)
        
        # Causal config
        self.causal_delay = self.reality_config.get('causal_delay_embedding', 3)
        self.causal_horizon = self.reality_config.get('causal_prediction_horizon', 1)
        
        logger.info("RealityEngine initialized")
    
    def compute_multiscale_entropy(self, frames: np.ndarray) -> Dict[str, float]:
        """
        Tính multi-scale Shannon entropy từ Gaussian pyramid
        
        Real video có entropy distribution tự nhiên qua các scales
        AI-generated có thể có entropy không nhất quán
        
        Args:
            frames: Array frames normalized
            
        Returns:
            Dictionary entropy features
        """
        features = {}
        
        # Convert to grayscale
        if frames.shape[-1] == 3:
            gray_frames = np.mean(frames, axis=-1)
        else:
            gray_frames = frames.squeeze(-1)
        
        # Tính entropy cho mỗi scale
        entropies_per_scale = []
        
        for scale_idx in range(self.entropy_scales):
            scale_entropies = []
            
            for frame in gray_frames:
                # Downsample frame theo scale
                downsampled = frame
                for _ in range(scale_idx):
                    downsampled = cv2.pyrDown(downsampled)
                
                # Convert to uint8 để tính histogram
                frame_uint8 = (downsampled * 255).astype(np.uint8)
                
                # Tính histogram
                hist, _ = np.histogram(frame_uint8, bins=256, range=(0, 256))
                
                # Normalize histogram
                hist = hist.astype(float) / (hist.sum() + 1e-10)
                
                # Shannon entropy
                entropy = -np.sum(hist * np.log2(hist + 1e-10))
                scale_entropies.append(entropy)
            
            entropies_per_scale.append(np.mean(scale_entropies))
        
        # Features từ entropy cross scales
        features['entropy_mean'] = float(np.mean(entropies_per_scale))
        features['entropy_std'] = float(np.std(entropies_per_scale))
        
        # Slope của entropy qua scales (real video thường giảm dần)
        if len(entropies_per_scale) > 1:
            x = np.arange(len(entropies_per_scale))
            slope = np.polyfit(x, entropies_per_scale, 1)[0]
            features['entropy_slope'] = float(slope)
        else:
            features['entropy_slope'] = 0.0
        
        logger.debug(f"Entropy features: {features}")
        return features
    
    def compute_fractal_dimension(self, frames: np.ndarray) -> Dict[str, float]:
        """
        Tính fractal dimension proxy bằng box-counting trên edge maps
        
        Real scenes có cấu trúc fractal tự nhiên
        AI-generated có thể thiếu self-similarity
        
        Args:
            frames: Array frames normalized
            
        Returns:
            Dictionary fractal features
        """
        features = {}
        
        # Convert to grayscale
        if frames.shape[-1] == 3:
            gray_frames = (frames * 255).astype(np.uint8)
            gray_frames = np.array([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in gray_frames])
        else:
            gray_frames = (frames.squeeze(-1) * 255).astype(np.uint8)
        
        fractal_dims = []
        
        for frame in gray_frames:
            # Detect edges bằng Canny
            edges = cv2.Canny(frame, self.edge_low, self.edge_high)
            
            # Box counting
            counts = []
            sizes = []
            
            for box_size in self.fractal_boxes:
                # Downsample edges theo box_size
                h, w = edges.shape
                h_new = h // box_size
                w_new = w // box_size
                
                if h_new < 2 or w_new < 2:
                    continue
                
                # Reshape và count boxes có edge
                boxes = edges[:h_new*box_size, :w_new*box_size].reshape(
                    h_new, box_size, w_new, box_size
                )
                
                # Count boxes có ít nhất 1 edge pixel
                box_has_edge = (boxes.sum(axis=(1, 3)) > 0).sum()
                
                counts.append(box_has_edge)
                sizes.append(box_size)
            
            # Fit line: log(count) vs log(1/size)
            if len(counts) > 2:
                log_counts = np.log(np.array(counts) + 1)
                log_sizes = np.log(1.0 / np.array(sizes))
                
                # Slope = fractal dimension estimate
                slope = np.polyfit(log_sizes, log_counts, 1)[0]
                fractal_dims.append(slope)
        
        if len(fractal_dims) > 0:
            features['fractal_dim_mean'] = float(np.mean(fractal_dims))
            features['fractal_dim_std'] = float(np.std(fractal_dims))
        else:
            features['fractal_dim_mean'] = 0.0
            features['fractal_dim_std'] = 0.0
        
        logger.debug(f"Fractal features: {features}")
        return features
    
    def compute_causal_motion(self, frames: np.ndarray) -> Dict[str, float]:
        """
        Dự đoán chuyển động nhân quả bằng delay embedding + linear prediction
        
        Real motion tuân theo vật lý → dự đoán được
        AI-generated có thể có jumps không tự nhiên
        
        Args:
            frames: Array frames normalized
            
        Returns:
            Dictionary causal motion features
        """
        features = {}
        
        # Tính motion statistics qua time
        if frames.shape[-1] == 3:
            gray_frames = np.mean(frames, axis=-1)
        else:
            gray_frames = frames.squeeze(-1)
        
        # Simple motion metric: mean absolute difference giữa frames
        motion_series = []
        for i in range(len(gray_frames) - 1):
            diff = np.abs(gray_frames[i+1] - gray_frames[i])
            motion_series.append(np.mean(diff))
        
        motion_series = np.array(motion_series)
        
        if len(motion_series) < self.causal_delay + self.causal_horizon + 1:
            # Không đủ frames để dự đoán
            features['causal_prediction_error'] = 0.0
            features['causal_predictability'] = 0.0
            return features
        
        # Delay embedding
        X = []
        y = []
        
        for i in range(len(motion_series) - self.causal_delay - self.causal_horizon):
            # Features: [t, t+1, ..., t+delay-1]
            X.append(motion_series[i:i+self.causal_delay])
            # Target: t+delay+horizon
            y.append(motion_series[i+self.causal_delay+self.causal_horizon-1])
        
        X = np.array(X)
        y = np.array(y)
        
        # Linear prediction model (Ridge regression)
        model = Ridge(alpha=0.1)
        
        # Simple train/test split
        split = int(0.7 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        if len(X_train) == 0 or len(X_test) == 0:
            features['causal_prediction_error'] = 0.0
            features['causal_predictability'] = 0.0
            return features
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Prediction error
        mse = np.mean((y_pred - y_test) ** 2)
        features['causal_prediction_error'] = float(mse)
        
        # Predictability score (R^2)
        score = model.score(X_test, y_test)
        features['causal_predictability'] = float(max(0.0, score))
        
        logger.debug(f"Causal motion features: {features}")
        return features
    
    def compute_information_conservation(self, frames: np.ndarray) -> Dict[str, float]:
        """
        Kiểm tra information conservation qua compression và complexity
        
        Real video có compression delta tự nhiên
        AI-generated có thể có patterns lặp hoặc complexity bất thường
        
        Args:
            frames: Array frames normalized
            
        Returns:
            Dictionary information conservation features
        """
        features = {}
        
        # Tính compression proxy bằng gzip trên frame data
        compression_ratios = []
        
        frames_uint8 = (frames * 255).astype(np.uint8)
        
        for frame in frames_uint8:
            # Flatten frame
            flat = frame.flatten()
            
            # Gzip compression
            import gzip
            original_size = len(flat.tobytes())
            compressed = gzip.compress(flat.tobytes(), compresslevel=6)
            compressed_size = len(compressed)
            
            ratio = safe_divide(compressed_size, original_size, default=1.0)
            compression_ratios.append(ratio)
        
        compression_ratios = np.array(compression_ratios)
        
        # Compression statistics
        features['compression_mean'] = float(np.mean(compression_ratios))
        features['compression_std'] = float(np.std(compression_ratios))
        
        # Compression delta (thay đổi compression qua time)
        if len(compression_ratios) > 1:
            deltas = np.diff(compression_ratios)
            features['compression_delta_mean'] = float(np.mean(np.abs(deltas)))
        else:
            features['compression_delta_mean'] = 0.0
        
        # Frame complexity proxy: std của pixel values
        complexities = [np.std(frame) for frame in frames]
        features['complexity_mean'] = float(np.mean(complexities))
        
        logger.debug(f"Information conservation features: {features}")
        return features
    
    def analyze(self, frames: np.ndarray) -> Dict[str, Any]:
        """
        Reality compliance analysis đầy đủ
        
        Args:
            frames: Array frames normalized
            
        Returns:
            Dictionary tổng hợp reality features
        """
        logger.info("Bắt đầu reality compliance analysis...")
        
        all_features = {}
        
        # Multi-scale entropy
        entropy_feats = self.compute_multiscale_entropy(frames)
        all_features.update(entropy_feats)
        
        # Fractal dimension
        fractal_feats = self.compute_fractal_dimension(frames)
        all_features.update(fractal_feats)
        
        # Causal motion
        causal_feats = self.compute_causal_motion(frames)
        all_features.update(causal_feats)
        
        # Information conservation
        conservation_feats = self.compute_information_conservation(frames)
        all_features.update(conservation_feats)
        
        logger.info(f"Reality analysis hoàn tất, {len(all_features)} features")
        return all_features