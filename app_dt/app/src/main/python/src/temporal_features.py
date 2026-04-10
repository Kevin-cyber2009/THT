"""
Temporal Features Module for AI Video Detection
================================================

Extracts temporal/sequential features that detect AI-generated video artifacts.

Features:
- Optical flow consistency
- Frame-to-frame change patterns
- Temporal frequency analysis
- Motion smoothness
- Noise temporal consistency
- Compression artifacts over time
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy import signal
from scipy.fft import fft, fftfreq

logger = logging.getLogger('hybrid_detector.temporal')


class TemporalAnalyzer:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        
        self.temporal_config = self.config.get('temporal', {})
        self.window_size = self.temporal_config.get('window_size', 5)
        self.motion_threshold = self.temporal_config.get('motion_threshold', 0.5)
        
        self.frame_history: List[np.ndarray] = []
        self.flow_history: List[np.ndarray] = []
        self.diff_history: List[float] = []
        
        logger.info("TemporalAnalyzer initialized")

    def compute_frame_differences(self, frames: np.ndarray) -> Dict[str, float]:
        """Compute frame-to-frame differences."""
        features = {}
        
        if len(frames) < 2:
            return self._default_diff_features()
        
        gray_frames = []
        for frame in frames:
            if frame.shape[-1] == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.squeeze()
            gray_frames.append(gray.astype(np.float32))
        
        differences = []
        for i in range(len(gray_frames) - 1):
            diff = cv2.absdiff(gray_frames[i], gray_frames[i + 1])
            mean_diff = float(np.mean(diff))
            differences.append(mean_diff)
            self.diff_history.append(mean_diff)
        
        if self.diff_history:
            self.diff_history = self.diff_history[-100:]
        
        features['frame_diff_mean'] = float(np.mean(differences)) if differences else 0.0
        features['frame_diff_std'] = float(np.std(differences)) if differences else 0.0
        features['frame_diff_max'] = float(np.max(differences)) if differences else 0.0
        features['frame_diff_min'] = float(np.min(differences)) if differences else 0.0
        
        if len(differences) > 1:
            features['frame_diff_trend'] = float(np.polyfit(range(len(differences)), differences, 1)[0])
        else:
            features['frame_diff_trend'] = 0.0
        
        return features

    def compute_optical_flow_features(self, frames: np.ndarray) -> Dict[str, float]:
        """Compute optical flow features."""
        features = {}
        
        if len(frames) < 2:
            return self._default_flow_features()
        
        gray_frames = []
        for frame in frames:
            if frame.shape[-1] == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.squeeze()
            gray_frames.append(gray)
        
        flow_magnitudes = []
        flow_angles = []
        flow_smoothness = []
        
        for i in range(len(gray_frames) - 1):
            flow = cv2.calcOpticalFlowFarneback(
                gray_frames[i],
                gray_frames[i + 1],
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
            
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_magnitudes.append(mag)
            flow_angles.append(ang)
            self.flow_history.append(np.mean(mag))
            
            grad_x = np.gradient(mag, axis=1)
            grad_y = np.gradient(mag, axis=0)
            smoothness = 1.0 / (1.0 + np.mean(grad_x**2 + grad_y**2))
            flow_smoothness.append(smoothness)
        
        if self.flow_history:
            self.flow_history = self.flow_history[-100:]
        
        if flow_magnitudes:
            mag_stack = np.array(flow_magnitudes)
            features['flow_magnitude_mean'] = float(np.mean(mag_stack))
            features['flow_magnitude_std'] = float(np.std(mag_stack))
            features['flow_magnitude_max'] = float(np.max(mag_stack))
            
            features['flow_smoothness_mean'] = float(np.mean(flow_smoothness))
            features['flow_smoothness_std'] = float(np.std(flow_smoothness))
            
            if len(flow_magnitudes) > 1:
                corrs = []
                for i in range(len(mag_stack) - 1):
                    a = mag_stack[i].flatten()
                    b = mag_stack[i + 1].flatten()
                    min_len = min(len(a), len(b))
                    c = np.corrcoef(a[:min_len], b[:min_len])[0, 1]
                    if not np.isnan(c):
                        corrs.append(c)
                features['flow_temporal_correlation'] = float(np.mean(corrs)) if corrs else 0.0
            else:
                features['flow_temporal_correlation'] = 0.0
        else:
            features.update(self._default_flow_features())
        
        return features

    def compute_temporal_frequency_features(self, frames: np.ndarray) -> Dict[str, float]:
        """Compute frequency domain features."""
        features = {}
        
        if len(frames) < 8:
            return self._default_freq_features()
        
        gray_frames = []
        for frame in frames:
            if frame.shape[-1] == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.squeeze()
            gray_frames.append(gray)
        
        temporal_signals = []
        for y in range(0, gray_frames[0].shape[0], 32):
            for x in range(0, gray_frames[0].shape[1], 32):
                signal = [f[y, x] for f in gray_frames]
                temporal_signals.append(signal)
        
        if not temporal_signals:
            return self._default_freq_features()
        
        fft_energies = []
        for signal in temporal_signals:
            fft_vals = np.abs(fft(signal))
            fft_freqs = fftfreq(len(signal), 1)
            
            positive_freqs = fft_freqs[:len(fft_freqs)//2]
            positive_energies = fft_vals[:len(fft_vals)//2]
            
            low_freq_energy = np.sum(positive_energies[positive_freqs < 0.2])
            high_freq_energy = np.sum(positive_energies[positive_freqs >= 0.2])
            
            if low_freq_energy > 0:
                fft_energies.append(high_freq_energy / low_freq_energy)
        
        features['temporal_fft_ratio'] = float(np.mean(fft_energies)) if fft_energies else 0.0
        features['temporal_fft_ratio_std'] = float(np.std(fft_energies)) if fft_energies else 0.0
        
        return features

    def compute_motion_smoothness(self, frames: np.ndarray) -> Dict[str, float]:
        """Compute motion smoothness features."""
        features = {}
        
        if len(frames) < 3:
            return self._default_motion_features()
        
        gray_frames = []
        for frame in frames:
            if frame.shape[-1] == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.squeeze()
            gray_frames.append(gray.astype(np.float32))
        
        motion_vectors = []
        for i in range(len(gray_frames) - 1):
            diff = gray_frames[i + 1].astype(np.float32) - gray_frames[i].astype(np.float32)
            motion_vectors.append(np.mean(np.abs(diff)))
        
        if len(motion_vectors) < 2:
            return self._default_motion_features()
        
        motion_array = np.array(motion_vectors)
        acceleration = np.diff(motion_array)
        
        features['motion_mean'] = float(np.mean(motion_array))
        features['motion_std'] = float(np.std(motion_array))
        features['motion_smoothness'] = 1.0 / (1.0 + np.mean(np.abs(acceleration)))
        features['motion_jerkiness'] = float(np.mean(np.abs(acceleration)))
        
        if len(motion_array) > 1:
            corrs = []
            for i in range(len(motion_array) - 1):
                c = np.corrcoef(motion_array[i:i+2])[0, 1]
                if not np.isnan(c):
                    corrs.append(c)
            features['motion_temporal_consistency'] = float(np.mean(corrs)) if corrs else 0.0
        else:
            features['motion_temporal_consistency'] = 0.0
        
        return features

    def compute_noise_temporal_features(self, frames: np.ndarray) -> Dict[str, float]:
        """Compute noise consistency features."""
        features = {}
        
        if len(frames) < 2:
            return self._default_noise_features()
        
        noise_estimates = []
        
        for i in range(len(frames) - 1):
            frame1 = frames[i]
            frame2 = frames[i + 1]
            
            if frame1.shape != frame2.shape:
                continue
            
            if frame1.shape[-1] == 3:
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            else:
                gray1 = frame1.squeeze()
                gray2 = frame2.squeeze()
            
            diff = cv2.absdiff(gray1.astype(np.float32), gray2.astype(np.float32))
            
            noise_est = float(np.median(diff) / 0.6745)
            noise_estimates.append(noise_est)
        
        if noise_estimates:
            features['noise_estimate_mean'] = float(np.mean(noise_estimates))
            features['noise_estimate_std'] = float(np.std(noise_estimates))
            features['noise_temporal_variance'] = float(np.var(noise_estimates))
        else:
            features.update(self._default_noise_features())
        
        return features

    def compute_compression_artifact_features(self, frames: np.ndarray) -> Dict[str, float]:
        """Compute compression artifact features."""
        features = {}
        
        if len(frames) < 2:
            return self._default_compression_features()
        
        artifact_scores = []
        
        for frame in frames:
            if frame.shape[-1] == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.squeeze()
            
            dct = cv2.dct(gray.astype(np.float32))
            
            high_freq_energy = np.sum(dct[8:, 8:] ** 2)
            total_energy = np.sum(dct ** 2)
            
            if total_energy > 0:
                artifact_ratio = high_freq_energy / total_energy
                artifact_scores.append(artifact_ratio)
        
        if artifact_scores:
            features['compression_artifact_ratio_mean'] = float(np.mean(artifact_scores))
            features['compression_artifact_ratio_std'] = float(np.std(artifact_scores))
            features['compression_artifact_ratio_trend'] = float(
                np.polyfit(range(len(artifact_scores)), artifact_scores, 1)[0]
            ) if len(artifact_scores) > 1 else 0.0
        else:
            features.update(self._default_compression_features())
        
        return features

    def extract_all_features(self, frames: np.ndarray) -> Dict[str, Any]:
        """Extract all temporal features."""
        logger.info(f"Extracting temporal features from {len(frames)} frames...")
        
        if len(frames.shape) == 4:
            frames_list = [frames[i] for i in range(len(frames))]
        else:
            frames_list = [frames]
        
        features = {}
        
        diff_features = self.compute_frame_differences(np.array(frames_list))
        features.update(diff_features)
        
        flow_features = self.compute_optical_flow_features(np.array(frames_list))
        features.update(flow_features)
        
        freq_features = self.compute_temporal_frequency_features(np.array(frames_list))
        features.update(freq_features)
        
        motion_features = self.compute_motion_smoothness(np.array(frames_list))
        features.update(motion_features)
        
        noise_features = self.compute_noise_temporal_features(np.array(frames_list))
        features.update(noise_features)
        
        compression_features = self.compute_compression_artifact_features(np.array(frames_list))
        features.update(compression_features)
        
        logger.info(f"Temporal analysis complete: {len(features)} features extracted")
        
        return features

    def _default_diff_features(self) -> Dict[str, float]:
        return {
            'frame_diff_mean': 5.0,
            'frame_diff_std': 2.0,
            'frame_diff_max': 15.0,
            'frame_diff_min': 1.0,
            'frame_diff_trend': 0.0,
        }

    def _default_flow_features(self) -> Dict[str, float]:
        return {
            'flow_magnitude_mean': 2.0,
            'flow_magnitude_std': 1.0,
            'flow_magnitude_max': 10.0,
            'flow_smoothness_mean': 0.5,
            'flow_smoothness_std': 0.1,
            'flow_temporal_correlation': 0.5,
        }

    def _default_freq_features(self) -> Dict[str, float]:
        return {
            'temporal_fft_ratio': 0.1,
            'temporal_fft_ratio_std': 0.05,
        }

    def _default_motion_features(self) -> Dict[str, float]:
        return {
            'motion_mean': 5.0,
            'motion_std': 2.0,
            'motion_smoothness': 0.5,
            'motion_jerkiness': 1.0,
            'motion_temporal_consistency': 0.5,
        }

    def _default_noise_features(self) -> Dict[str, float]:
        return {
            'noise_estimate_mean': 2.0,
            'noise_estimate_std': 0.5,
            'noise_temporal_variance': 0.5,
        }

    def _default_compression_features(self) -> Dict[str, float]:
        return {
            'compression_artifact_ratio_mean': 0.1,
            'compression_artifact_ratio_std': 0.02,
            'compression_artifact_ratio_trend': 0.0,
        }

    def reset(self):
        """Reset analyzer state."""
        self.frame_history = []
        self.flow_history = []
        self.diff_history = []
