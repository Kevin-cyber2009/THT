# src/fusion.py (sửa lại phần import và function)

"""
Module fusion: Kết hợp artifact/reality/stress scores thành final probability
"""

import numpy as np
from typing import Dict, Any, Optional, List
import logging

from .utils import load_config, clip_value


logger = logging.getLogger('hybrid_detector.fusion')


class ScoreFusion:
    """
    Class fusion các component scores thành final decision
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Khởi tạo ScoreFusion
        
        Args:
            config: Dictionary cấu hình
        """
        if config is None:
            config = load_config()
        
        self.fusion_config = config.get('fusion', {})
        
        # Weights cho các components
        self.artifact_weight = self.fusion_config.get('artifact_weight', 0.4)
        self.reality_weight = self.fusion_config.get('reality_weight', 0.35)
        self.stress_weight = self.fusion_config.get('stress_weight', 0.25)
        
        # Normalize weights
        total_weight = self.artifact_weight + self.reality_weight + self.stress_weight
        self.artifact_weight /= total_weight
        self.reality_weight /= total_weight
        self.stress_weight /= total_weight
        
        # Thresholds
        self.threshold_fake = self.fusion_config.get('threshold_fake', 0.5)
        self.confidence_high = self.fusion_config.get('confidence_threshold_high', 0.7)
        self.confidence_low = self.fusion_config.get('confidence_threshold_low', 0.3)
        
        logger.info(f"ScoreFusion initialized - weights: artifact={self.artifact_weight:.2f}, "
                   f"reality={self.reality_weight:.2f}, stress={self.stress_weight:.2f}")
    
    def compute_artifact_score(self, features: Dict[str, float]) -> float:
        """
        Tính artifact score từ forensic features
        
        Score cao → nhiều artifacts → có thể fake
        
        Args:
            features: Dictionary forensic features
            
        Returns:
            Artifact score [0, 1]
        """
        # Các indicators của artifacts
        indicators = []
        
        # FFT anomalies
        fft_std = features.get('fft_std', 0)
        if fft_std > 0.05:  # Threshold từ thực nghiệm
            indicators.append(1.0)
        else:
            indicators.append(0.0)
        
        # PRNU inconsistencies
        prnu_autocorr = features.get('prnu_autocorr', 0)
        if prnu_autocorr < 0.3:  # Low autocorr → không phải sensor thật
            indicators.append(1.0)
        else:
            indicators.append(0.0)
        
        # Optical flow unnaturalness
        flow_smoothness = features.get('flow_smoothness', 0)
        if flow_smoothness < 0.5:
            indicators.append(1.0)
        else:
            indicators.append(0.0)
        
        # DCT anomalies
        dct_ac_energy = features.get('dct_ac_energy', 0)
        if dct_ac_energy > 0.2:
            indicators.append(0.5)
        
        if len(indicators) > 0:
            score = np.mean(indicators)
        else:
            score = 0.5  # Default neutral
        
        return float(clip_value(score, 0, 1))
    
    def compute_reality_score(self, features: Dict[str, float]) -> float:
        """
        Tính reality compliance score
        
        Score cao → tuân thủ reality → có thể real
        Score thấp → vi phạm reality → có thể fake
        
        Args:
            features: Dictionary reality features
            
        Returns:
            Reality score [0, 1]
        """
        indicators = []
        
        # Entropy slope (real thường âm khi downsample)
        entropy_slope = features.get('entropy_slope', 0)
        if entropy_slope < -0.2:
            indicators.append(1.0)
        elif entropy_slope > 0.2:
            indicators.append(0.0)
        else:
            indicators.append(0.5)
        
        # Fractal dimension (real scenes ≈ 1.5-2.0)
        fractal_dim = features.get('fractal_dim_mean', 0)
        if 1.5 <= fractal_dim <= 2.0:
            indicators.append(1.0)
        else:
            indicators.append(0.3)
        
        # Causal predictability (real motion dự đoán được)
        causal_pred = features.get('causal_predictability', 0)
        if causal_pred > 0.5:
            indicators.append(1.0)
        else:
            indicators.append(0.0)
        
        # Compression consistency
        comp_delta = features.get('compression_delta_mean', 0)
        if comp_delta < 0.1:  # Stable compression
            indicators.append(1.0)
        else:
            indicators.append(0.5)
        
        if len(indicators) > 0:
            score = np.mean(indicators)
        else:
            score = 0.5
        
        return float(clip_value(score, 0, 1))
    
    def compute_stress_score(self, stress_results: Dict[str, Any]) -> float:
        """
        Tính stress stability score
        
        Score cao → stable under perturbations → có thể real
        Score thấp → unstable → có thể fake
        
        Args:
            stress_results: Results từ StressLab
            
        Returns:
            Stress score [0, 1]
        """
        aggregate_stability = stress_results.get('aggregate_stability_score', 0)
        
        # Convert stability thành score
        # High stability → high score → real
        score = aggregate_stability
        
        return float(clip_value(score, 0, 1))
    
    def fuse_scores(
        self,
        artifact_score: float,
        reality_score: float,
        stress_score: float
    ) -> Dict[str, Any]:
        """
        Kết hợp các scores thành final probability
        
        Args:
            artifact_score: Artifact score [0, 1]
            reality_score: Reality score [0, 1]
            stress_score: Stress score [0, 1]
            
        Returns:
            Dictionary chứa final probability và metadata
        """
        # Weighted combination
        # artifact_score cao → fake
        # reality_score cao → real
        # stress_score cao → real
        
        # Convert reality và stress thành "fakeness" scores
        reality_fake = 1.0 - reality_score
        stress_fake = 1.0 - stress_score
        
        # Weighted average
        final_prob = (
            self.artifact_weight * artifact_score +
            self.reality_weight * reality_fake +
            self.stress_weight * stress_fake
        )
        
        final_prob = clip_value(final_prob, 0, 1)
        
        # Decision
        if final_prob >= self.threshold_fake:
            prediction = "FAKE"
        else:
            prediction = "REAL"
        
        # Confidence level
        if final_prob >= self.confidence_high or final_prob <= (1 - self.confidence_high):
            confidence = "HIGH"
        elif final_prob >= self.confidence_low and final_prob <= (1 - self.confidence_low):
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        result = {
            'final_probability': float(final_prob),
            'artifact_score': float(artifact_score),
            'reality_score': float(reality_score),
            'stress_score': float(stress_score),
            'prediction': prediction,
            'confidence': confidence
        }
        
        logger.info(f"Fusion result: {prediction} (prob={final_prob:.3f}, confidence={confidence})")
        
        return result
    
    def generate_explanation(
        self,
        features: Dict[str, float],
        fusion_result: Dict[str, Any]
    ) -> List[str]:
        """
        Tạo explanation bullets cho kết quả
        
        Args:
            features: All features
            fusion_result: Kết quả từ fuse_scores
            
        Returns:
            List 3 explanation strings
        """
        explanations = []
        
        # Explanation 1: Forensic
        fft_std = features.get('fft_std', 0)
        prnu_autocorr = features.get('prnu_autocorr', 0)
        
        if fusion_result['artifact_score'] > 0.6:
            explanations.append(
                f"Phát hiện artifacts: Phổ FFT có độ lệch chuẩn cao ({fft_std:.3f}), "
                f"PRNU autocorrelation thấp ({prnu_autocorr:.3f})"
            )
        else:
            explanations.append(
                f"Phổ tần số FFT cho thấy phân bố tự nhiên ({fft_std:.3f}), "
                f"PRNU residual có tự tương quan cao ({prnu_autocorr:.3f})"
            )
        
        # Explanation 2: Reality
        entropy_slope = features.get('entropy_slope', 0)
        fractal_dim = features.get('fractal_dim_mean', 0)
        
        if fusion_result['reality_score'] > 0.6:
            explanations.append(
                f"Tuân thủ vật lý thực tại: Entropy slope {entropy_slope:.3f}, "
                f"fractal dimension {fractal_dim:.2f} trong khoảng tự nhiên"
            )
        else:
            explanations.append(
                f"Vi phạm reality compliance: Entropy slope bất thường ({entropy_slope:.3f}), "
                f"fractal dimension {fractal_dim:.2f}"
            )
        
        # Explanation 3: Stress
        if fusion_result['stress_score'] > 0.7:
            explanations.append(
                f"Ổn định cao dưới stress tests (stability {fusion_result['stress_score']:.2f}), "
                f"đặc trưng nhất quán khi bị nhiễu loạn"
            )
        else:
            explanations.append(
                f"Không ổn định dưới perturbations (stability {fusion_result['stress_score']:.2f}), "
                f"features thay đổi mạnh khi bị stress test"
            )
        
        return explanations[:3]  # Đảm bảo chỉ 3 bullets