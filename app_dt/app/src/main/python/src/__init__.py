"""
Hybrid Detector - AI Video Detection Package
============================================

Modules:
- features: Traditional feature extraction (FFT, DCT, PRNU, Optical Flow)
- fusion: Score fusion engine
- face_analyzer: Face-specific analysis (NEW)
- temporal_features: Temporal/sequential analysis (NEW)
- forensic: Forensic analysis
- reality_engine: Reality compliance checking
- stress_lab: Stress testing
- utils: Utility functions
"""

__version__ = "2.0.0"
__author__ = "AI Checker Team"

from .features import FeatureExtractor
from .fusion import ScoreFusion

__all__ = [
    'FeatureExtractor',
    'ScoreFusion',
]
