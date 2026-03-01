# src/stress_lab.py
"""
Module stress_lab: Adversarial stress testing với perturbations vật lý
"""

import cv2
import numpy as np
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

from .utils import load_config, safe_divide, clip_value


logger = logging.getLogger('hybrid_detector.stress_lab')


class StressLab:
    """
    Class thực hiện adversarial stress tests trên video
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Khởi tạo StressLab
        
        Args:
            config: Dictionary cấu hình
        """
        if config is None:
            config = load_config()
        
        self.stress_config = config.get('stress_lab', {})
        
        # Perturbation strengths
        self.light_jitter = self.stress_config.get('light_jitter_strength', 0.1)
        self.blur_range = self.stress_config.get('blur_kernel_range', [3, 7])
        self.rotation_range = self.stress_config.get('affine_rotation_range', [-5, 5])
        self.scale_range = self.stress_config.get('affine_scale_range', [0.95, 1.05])
        
        # Compression
        self.compression_levels = self.stress_config.get('compression_levels', [23, 28, 35])
        
        # Noise
        self.noise_std = self.stress_config.get('noise_std', 0.02)
        
        # Thresholds
        self.stability_low = self.stress_config.get('stability_threshold_low', 0.7)
        self.stability_high = self.stress_config.get('stability_threshold_high', 0.9)
        
        logger.info("StressLab initialized")
    
    def apply_light_jitter(self, frames: np.ndarray) -> np.ndarray:
        """
        Áp dụng jitter ánh sáng (brightness/contrast variation)
        
        Args:
            frames: Array frames normalized [0, 1]
            
        Returns:
            Perturbed frames
        """
        perturbed = frames.copy()
        
        for i in range(len(perturbed)):
            # Random brightness shift
            brightness_shift = np.random.uniform(-self.light_jitter, self.light_jitter)
            perturbed[i] = perturbed[i] + brightness_shift
            
            # Random contrast adjustment
            contrast_factor = np.random.uniform(1 - self.light_jitter, 1 + self.light_jitter)
            mean = np.mean(perturbed[i])
            perturbed[i] = (perturbed[i] - mean) * contrast_factor + mean
            
            # Clip to valid range
            perturbed[i] = np.clip(perturbed[i], 0, 1)
        
        return perturbed
    
    def apply_blur(self, frames: np.ndarray) -> np.ndarray:
        """
        Áp dụng Gaussian blur ngẫu nhiên
        
        Args:
            frames: Array frames normalized
            
        Returns:
            Blurred frames
        """
        perturbed = []
        
        for frame in frames:
            # Random kernel size (odd number)
            kernel_size = np.random.choice(range(self.blur_range[0], self.blur_range[1] + 1, 2))
            
            # Apply blur
            blurred = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
            perturbed.append(blurred)
        
        return np.array(perturbed)
    
    def apply_affine_jitter(self, frames: np.ndarray) -> np.ndarray:
        """
        Áp dụng small affine transformations (rotation, scale)
        
        Args:
            frames: Array frames normalized
            
        Returns:
            Transformed frames
        """
        perturbed = []
        h, w = frames.shape[1:3]
        center = (w // 2, h // 2)
        
        for frame in frames:
            # Random rotation
            angle = np.random.uniform(self.rotation_range[0], self.rotation_range[1])
            
            # Random scale
            scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
            
            # Affine matrix
            M = cv2.getRotationMatrix2D(center, angle, scale)
            
            # Apply transformation
            transformed = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            perturbed.append(transformed)
        
        return np.array(perturbed)
    
    def apply_noise(self, frames: np.ndarray) -> np.ndarray:
        """
        Thêm Gaussian noise
        
        Args:
            frames: Array frames normalized
            
        Returns:
            Noisy frames
        """
        noise = np.random.normal(0, self.noise_std, frames.shape)
        perturbed = frames + noise
        perturbed = np.clip(perturbed, 0, 1)
        
        return perturbed
    
    def apply_temporal_shuffle(self, frames: np.ndarray, window_size: int = 10) -> np.ndarray:
        """
        Shuffle frames trong windows nhỏ để test temporal consistency
        
        Args:
            frames: Array frames
            window_size: Kích thước window để shuffle
            
        Returns:
            Shuffled frames
        """
        perturbed = frames.copy()
        n_frames = len(frames)
        
        # Shuffle từng window
        for start in range(0, n_frames, window_size):
            end = min(start + window_size, n_frames)
            indices = np.arange(start, end)
            np.random.shuffle(indices)
            perturbed[start:end] = frames[indices]
        
        return perturbed
    
    def apply_compression_cascade(
        self, 
        frames: np.ndarray, 
        temp_dir: Optional[Path] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Áp dụng compression cascade test (nén nhiều lần)
        
        Args:
            frames: Array frames normalized
            temp_dir: Thư mục temporary để lưu video
            
        Returns:
            Tuple (compressed_frames, compression_stats)
        """
        if temp_dir is None:
            temp_dir = Path(tempfile.mkdtemp())
        
        temp_dir.mkdir(exist_ok=True)
        
        # Convert frames to video
        input_video = temp_dir / "input.mp4"
        self._frames_to_video(frames, str(input_video))
        
        compression_stats = {}
        current_video = input_video
        
        for i, crf in enumerate(self.compression_levels):
            output_video = temp_dir / f"compressed_{i}.mp4"
            
            # Compress with FFmpeg
            cmd = [
                'ffmpeg', '-y', '-i', str(current_video),
                '-c:v', 'libx264', '-crf', str(crf),
                '-preset', 'medium', '-an',
                str(output_video)
            ]
            
            try:
                subprocess.run(cmd, capture_output=True, check=True, timeout=30)
                
                # Get file sizes
                size_before = current_video.stat().st_size
                size_after = output_video.stat().st_size
                compression_stats[f'compression_ratio_{i}'] = safe_divide(size_after, size_before)
                
                current_video = output_video
                
            except subprocess.SubprocessError as e:
                logger.warning(f"Compression failed at level {crf}: {e}")
                break
        
        # Load final compressed frames
        compressed_frames = self._video_to_frames(str(current_video))
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return compressed_frames, compression_stats
    
    def _frames_to_video(self, frames: np.ndarray, output_path: str, fps: int = 6):
        """
        Convert frames array thành video file
        
        Args:
            frames: Array frames normalized [0, 1]
            output_path: Đường dẫn output video
            fps: Frame rate
        """
        frames_uint8 = (frames * 255).astype(np.uint8)
        h, w = frames.shape[1:3]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        for frame in frames_uint8:
            if frame.shape[-1] == 3:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                bgr = cv2.cvtColor(frame.squeeze(-1), cv2.COLOR_GRAY2BGR)
            out.write(bgr)
        
        out.release()
    
    def _video_to_frames(self, video_path: str) -> np.ndarray:
        """
        Đọc frames từ video
        
        Args:
            video_path: Đường dẫn video
            
        Returns:
            Array frames normalized
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_norm = frame_rgb.astype(np.float32) / 255.0
            frames.append(frame_norm)
        
        cap.release()
        
        return np.array(frames)
    
    def compute_stability_score(
        self, 
        features_original: Dict[str, float],
        features_perturbed: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Tính stability metrics: so sánh features trước và sau perturbation
        
        Args:
            features_original: Features từ video gốc
            features_perturbed: Features từ video đã perturb
            
        Returns:
            Dictionary stability ratios
        """
        stability = {}
        
        for key in features_original:
            if key in features_perturbed:
                orig_val = features_original[key]
                pert_val = features_perturbed[key]
                
                # Stability ratio (cần xử lý division by zero)
                if orig_val != 0:
                    ratio = pert_val / orig_val
                else:
                    ratio = 1.0 if pert_val == 0 else 0.0
                
                stability[f'stability_{key}'] = float(clip_value(ratio, 0, 2))
        
        # Overall stability score (mean của các ratios gần 1.0)
        ratios = list(stability.values())
        if len(ratios) > 0:
            # Tính độ lệch từ 1.0
            deviations = [abs(r - 1.0) for r in ratios]
            stability['overall_stability'] = float(1.0 - np.mean(deviations))
        else:
            stability['overall_stability'] = 0.0
        
        return stability
    
    def run_stress_tests(
        self, 
        frames: np.ndarray,
        feature_extractor
    ) -> Dict[str, Any]:
        """
        Chạy toàn bộ stress tests
        
        Args:
            frames: Array frames gốc
            feature_extractor: Object có method extract_features(frames)
            
        Returns:
            Dictionary chứa stability scores và metrics
        """
        logger.info("Bắt đầu stress tests...")
        
        # Extract features từ original
        original_features = feature_extractor.extract_features(frames)
        
        results = {
            'original_features': original_features,
            'perturbations': {}
        }
        
        # Test 1: Light jitter
        try:
            perturbed = self.apply_light_jitter(frames)
            pert_features = feature_extractor.extract_features(perturbed)
            stability = self.compute_stability_score(original_features, pert_features)
            results['perturbations']['light_jitter'] = stability
            logger.info(f"Light jitter stability: {stability.get('overall_stability', 0):.3f}")
        except Exception as e:
            logger.error(f"Light jitter test failed: {e}")
        
        # Test 2: Blur
        try:
            perturbed = self.apply_blur(frames)
            pert_features = feature_extractor.extract_features(perturbed)
            stability = self.compute_stability_score(original_features, pert_features)
            results['perturbations']['blur'] = stability
            logger.info(f"Blur stability: {stability.get('overall_stability', 0):.3f}")
        except Exception as e:
            logger.error(f"Blur test failed: {e}")
        
        # Test 3: Affine jitter
        try:
            perturbed = self.apply_affine_jitter(frames)
            pert_features = feature_extractor.extract_features(perturbed)
            stability = self.compute_stability_score(original_features, pert_features)
            results['perturbations']['affine'] = stability
            logger.info(f"Affine stability: {stability.get('overall_stability', 0):.3f}")
        except Exception as e:
            logger.error(f"Affine test failed: {e}")
        
        # Test 4: Noise
        try:
            perturbed = self.apply_noise(frames)
            pert_features = feature_extractor.extract_features(perturbed)
            stability = self.compute_stability_score(original_features, pert_features)
            results['perturbations']['noise'] = stability
            logger.info(f"Noise stability: {stability.get('overall_stability', 0):.3f}")
        except Exception as e:
            logger.error(f"Noise test failed: {e}")
        
        # Test 5: Temporal shuffle (chỉ test một số features nhanh)
        try:
            perturbed = self.apply_temporal_shuffle(frames)
            # Extract một subset features để nhanh hơn
            pert_features = feature_extractor.extract_features(perturbed)
            stability = self.compute_stability_score(original_features, pert_features)
            results['perturbations']['temporal_shuffle'] = stability
            logger.info(f"Temporal shuffle stability: {stability.get('overall_stability', 0):.3f}")
        except Exception as e:
            logger.error(f"Temporal shuffle test failed: {e}")
        
        # Aggregate stability score
        all_stabilities = []
        for pert_name, pert_results in results['perturbations'].items():
            if 'overall_stability' in pert_results:
                all_stabilities.append(pert_results['overall_stability'])
        
        if len(all_stabilities) > 0:
            results['aggregate_stability_score'] = float(np.mean(all_stabilities))
        else:
            results['aggregate_stability_score'] = 0.0
        
        logger.info(f"Stress tests hoàn tất, aggregate stability: {results['aggregate_stability_score']:.3f}")
        
        return results