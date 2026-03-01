# src/__init__.py
"""
Hybrid++ Reality Stress Video AI Detector
Package chính cho hệ thống phát hiện video AI-generated
"""

__version__ = "1.0.0"
__author__ = "Student Research Project"

from . import preprocessing
from . import forensic
from . import reality_engine
from . import stress_lab
from . import features
from . import classifier
from . import fusion
# from . import report  # Comment out vì chưa có file report.py
from . import utils

__all__ = [
    "preprocessing",
    "forensic",
    "reality_engine",
    "stress_lab",
    "features",
    "classifier",
    "fusion",
    # "report",  # Comment out
    "utils",
]