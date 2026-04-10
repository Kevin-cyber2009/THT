import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .utils import clip_value, load_config


logger = logging.getLogger('hybrid_detector.fusion')


class ScoreFusion:
    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = load_config()

        self.fusion_config = config.get('fusion', {})
        self.artifact_weight = self.fusion_config.get('artifact_weight', 0.4)
        self.reality_weight = self.fusion_config.get('reality_weight', 0.35)
        self.stress_weight = self.fusion_config.get('stress_weight', 0.25)

        total_weight = self.artifact_weight + self.reality_weight + self.stress_weight
        self.artifact_weight /= total_weight
        self.reality_weight /= total_weight
        self.stress_weight /= total_weight

        self.threshold_fake = self.fusion_config.get('threshold_fake', 0.5)
        self.confidence_high = self.fusion_config.get('confidence_threshold_high', 0.7)
        self.confidence_low = self.fusion_config.get('confidence_threshold_low', 0.3)

    @staticmethod
    def _normalize(value: float, low: float, high: float, invert: bool = False) -> float:
        if high <= low:
            return 0.5
        score = (float(value) - low) / (high - low)
        score = clip_value(score, 0.0, 1.0)
        return float(1.0 - score if invert else score)

    def compute_artifact_score(self, features: Dict[str, float]) -> float:
        indicators = [
            self._normalize(features.get('fft_std', 0.0), 0.02, 0.12),
            self._normalize(features.get('prnu_autocorr', 0.0), 0.15, 0.65, invert=True),
            self._normalize(features.get('flow_smoothness', 0.0), 0.25, 0.85, invert=True),
            self._normalize(features.get('dct_ac_energy', 0.0), 0.05, 0.45),
            self._normalize(features.get('fft_high_freq_energy', 0.0), 0.10, 0.90),
        ]
        return float(clip_value(np.mean(indicators) if indicators else 0.5, 0.0, 1.0))

    def compute_reality_score(self, features: Dict[str, float]) -> float:
        entropy_slope = features.get('entropy_slope', 0.0)
        entropy_score = 1.0 - min(abs(entropy_slope) / 0.35, 1.0)

        fractal_dim = features.get('fractal_dim_mean', 0.0)
        fractal_score = 1.0 - min(abs(fractal_dim - 1.75) / 0.45, 1.0)

        indicators = [
            entropy_score,
            fractal_score,
            self._normalize(features.get('causal_predictability', 0.0), 0.05, 0.75),
            self._normalize(features.get('compression_delta_mean', 0.0), 0.02, 0.20, invert=True),
            self._normalize(features.get('complexity_mean', 0.0), 0.03, 0.30),
        ]
        return float(clip_value(np.mean(indicators) if indicators else 0.5, 0.0, 1.0))

    def compute_stress_score(self, stress_results: Dict[str, Any]) -> float:
        return float(clip_value(stress_results.get('aggregate_stability_score', 0.0), 0.0, 1.0))

    def compute_stress_proxy(self, features: Dict[str, float]) -> float:
        indicators = [
            self._normalize(features.get('flow_temporal_consistency', 0.0), 0.0, 0.95),
            self._normalize(features.get('prnu_temporal_consistency', 0.0), 0.0, 0.08, invert=True),
            self._normalize(features.get('flow_std_magnitude', 0.0), 0.02, 0.60, invert=True),
        ]
        return float(clip_value(np.mean(indicators) if indicators else 0.5, 0.0, 1.0))

    def fuse_scores(self, artifact_score: float, reality_score: float, stress_score: float) -> Dict[str, Any]:
        final_prob = (
            self.artifact_weight * artifact_score
            + self.reality_weight * (1.0 - reality_score)
            + self.stress_weight * (1.0 - stress_score)
        )
        final_prob = clip_value(final_prob, 0.0, 1.0)
        prediction = "FAKE" if final_prob >= self.threshold_fake else "REAL"

        distance = abs(final_prob - 0.5)
        confidence = "HIGH" if distance >= 0.30 else "MEDIUM" if distance >= 0.15 else "LOW"
        return {
            'final_probability': float(final_prob),
            'artifact_score': float(artifact_score),
            'reality_score': float(reality_score),
            'stress_score': float(stress_score),
            'prediction': prediction,
            'confidence': confidence,
        }

    def generate_explanation(self, features: Dict[str, float], fusion_result: Dict[str, Any]) -> List[str]:
        explanations = []
        fft_std = features.get('fft_std', 0.0)
        prnu_autocorr = features.get('prnu_autocorr', 0.0)
        if fusion_result['artifact_score'] > 0.6:
            explanations.append(f"Artifact risk cao: FFT std={fft_std:.3f}, PRNU autocorr={prnu_autocorr:.3f}")
        else:
            explanations.append(f"Artifact risk thap: FFT/PRNU gan mau tu nhien ({fft_std:.3f}, {prnu_autocorr:.3f})")

        entropy_slope = features.get('entropy_slope', 0.0)
        fractal_dim = features.get('fractal_dim_mean', 0.0)
        if fusion_result['reality_score'] > 0.6:
            explanations.append(f"Reality compliance tot: entropy slope={entropy_slope:.3f}, fractal={fractal_dim:.2f}")
        else:
            explanations.append(f"Reality compliance yeu: entropy/fractal bat thuong ({entropy_slope:.3f}, {fractal_dim:.2f})")

        if fusion_result['stress_score'] > 0.7:
            explanations.append(f"On dinh tot duoi bien dong nhe, stress score={fusion_result['stress_score']:.2f}")
        else:
            explanations.append(f"Do on dinh thap khi co nhieu dao dong, stress score={fusion_result['stress_score']:.2f}")

        return explanations[:3]
