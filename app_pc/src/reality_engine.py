import cv2
import numpy as np
from scipy import ndimage
from sklearn.linear_model import Ridge
from typing import Dict, Any, Optional
import logging

from .utils import load_config, safe_divide


logger = logging.getLogger('hybrid_detector.reality_engine')


class RealityEngine:
    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = load_config()
        
        self.reality_config = config.get('reality_engine', {})
        
        self.entropy_scales = self.reality_config.get('entropy_scales', 4)
        self.entropy_window = self.reality_config.get('entropy_window_size', 16)
        
        self.fractal_boxes = self.reality_config.get('fractal_box_sizes', [2, 4, 8, 16, 32])
        self.edge_low = self.reality_config.get('fractal_edge_threshold_low', 50)
        self.edge_high = self.reality_config.get('fractal_edge_threshold_high', 150)
        
        self.causal_delay = self.reality_config.get('causal_delay_embedding', 3)
        self.causal_horizon = self.reality_config.get('causal_prediction_horizon', 1)
        
        logger.info("RealityEngine initialized")
    
    def compute_multiscale_entropy(self, frames: np.ndarray) -> Dict[str, float]:
        features = {}
        
        if frames.shape[-1] == 3:
            gray_frames = np.mean(frames, axis=-1)
        else:
            gray_frames = frames.squeeze(-1)
        
        entropies_per_scale = []
        
        for scale_idx in range(self.entropy_scales):
            scale_entropies = []
            
            for frame in gray_frames:
                downsampled = frame
                for _ in range(scale_idx):
                    downsampled = cv2.pyrDown(downsampled)
                
                frame_uint8 = (downsampled * 255).astype(np.uint8)
                
                hist, _ = np.histogram(frame_uint8, bins=256, range=(0, 256))
                
                hist = hist.astype(float) / (hist.sum() + 1e-10)
                
                entropy = -np.sum(hist * np.log2(hist + 1e-10))
                scale_entropies.append(entropy)
            
            entropies_per_scale.append(np.mean(scale_entropies))
        
        features['entropy_mean'] = float(np.mean(entropies_per_scale))
        features['entropy_std'] = float(np.std(entropies_per_scale))
        
        if len(entropies_per_scale) > 1:
            x = np.arange(len(entropies_per_scale))
            slope = np.polyfit(x, entropies_per_scale, 1)[0]
            features['entropy_slope'] = float(slope)
        else:
            features['entropy_slope'] = 0.0
        
        logger.debug(f"Entropy features: {features}")
        return features
    
    def compute_fractal_dimension(self, frames: np.ndarray) -> Dict[str, float]:
        features = {}
        
        if frames.shape[-1] == 3:
            gray_frames = (frames * 255).astype(np.uint8)
            gray_frames = np.array([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in gray_frames])
        else:
            gray_frames = (frames.squeeze(-1) * 255).astype(np.uint8)
        
        fractal_dims = []
        
        for frame in gray_frames:
            edges = cv2.Canny(frame, self.edge_low, self.edge_high)
            
            counts = []
            sizes = []
            
            for box_size in self.fractal_boxes:
                h, w = edges.shape
                h_new = h // box_size
                w_new = w // box_size
                
                if h_new < 2 or w_new < 2:
                    continue
                
                boxes = edges[:h_new*box_size, :w_new*box_size].reshape(
                    h_new, box_size, w_new, box_size
                )
                
                box_has_edge = (boxes.sum(axis=(1, 3)) > 0).sum()
                
                counts.append(box_has_edge)
                sizes.append(box_size)
            
            if len(counts) > 2:
                log_counts = np.log(np.array(counts) + 1)
                log_sizes = np.log(1.0 / np.array(sizes))
                
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
        features = {}
        
        if frames.shape[-1] == 3:
            gray_frames = np.mean(frames, axis=-1)
        else:
            gray_frames = frames.squeeze(-1)
        
        motion_series = []
        for i in range(len(gray_frames) - 1):
            diff = np.abs(gray_frames[i+1] - gray_frames[i])
            motion_series.append(np.mean(diff))
        
        motion_series = np.array(motion_series)
        
        if len(motion_series) < self.causal_delay + self.causal_horizon + 1:
            features['causal_prediction_error'] = 0.0
            features['causal_predictability'] = 0.0
            return features
        
        X = []
        y = []
        
        for i in range(len(motion_series) - self.causal_delay - self.causal_horizon):
            X.append(motion_series[i:i+self.causal_delay])
            y.append(motion_series[i+self.causal_delay+self.causal_horizon-1])
        
        X = np.array(X)
        y = np.array(y)
        
        model = Ridge(alpha=0.1)
        
        split = int(0.7 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        if len(X_train) == 0 or len(X_test) == 0:
            features['causal_prediction_error'] = 0.0
            features['causal_predictability'] = 0.0
            return features
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = np.mean((y_pred - y_test) ** 2)
        features['causal_prediction_error'] = float(mse)
        
        score = model.score(X_test, y_test)
        features['causal_predictability'] = float(max(0.0, score))
        
        logger.debug(f"Causal motion features: {features}")
        return features
    
    def compute_information_conservation(self, frames: np.ndarray) -> Dict[str, float]:
        features = {}
        
        compression_ratios = []
        
        frames_uint8 = (frames * 255).astype(np.uint8)
        
        for frame in frames_uint8:
            flat = frame.flatten()
            
            import gzip
            original_size = len(flat.tobytes())
            compressed = gzip.compress(flat.tobytes(), compresslevel=6)
            compressed_size = len(compressed)
            
            ratio = safe_divide(compressed_size, original_size, default=1.0)
            compression_ratios.append(ratio)
        
        compression_ratios = np.array(compression_ratios)
        
        features['compression_mean'] = float(np.mean(compression_ratios))
        features['compression_std'] = float(np.std(compression_ratios))
        
        if len(compression_ratios) > 1:
            deltas = np.diff(compression_ratios)
            features['compression_delta_mean'] = float(np.mean(np.abs(deltas)))
        else:
            features['compression_delta_mean'] = 0.0
        
        complexities = [np.std(frame) for frame in frames]
        features['complexity_mean'] = float(np.mean(complexities))
        
        logger.debug(f"Information conservation features: {features}")
        return features
    
    def analyze(self, frames: np.ndarray) -> Dict[str, Any]:
        logger.info("Bắt đầu reality compliance analysis...")
        
        all_features = {}
        
        entropy_feats = self.compute_multiscale_entropy(frames)
        all_features.update(entropy_feats)
        
        fractal_feats = self.compute_fractal_dimension(frames)
        all_features.update(fractal_feats)
        
        causal_feats = self.compute_causal_motion(frames)
        all_features.update(causal_feats)
        
        conservation_feats = self.compute_information_conservation(frames)
        all_features.update(conservation_feats)
        
        logger.info(f"Reality analysis hoàn tất, {len(all_features)} features")
        return all_features