# setup.py

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding='utf-8') if readme_file.exists() else ""

setup(
    name="hybrid-reality-detector",
    version="1.0.0",
    author="Student Research Project",
    author_email="your.email@example.com",
    description="Hybrid++ Reality Stress Video AI Detector - Phát hiện video AI-generated",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hybrid_reality_detector",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "lightgbm>=4.1.0",
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "pandas>=2.1.0",
        "PyYAML>=6.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "reportlab>=4.0.0",
        "PySide6>=6.6.0",
        "pytest>=7.4.0",
        "tqdm>=4.66.0",
        "joblib>=1.3.0",
    ],
    entry_points={
        'console_scripts': [
            'hybrid-detector=run_demo:main',
            'hybrid-train=train_classifier:main',
        ],
    },
    include_package_data=True,
    package_data={
        '': ['config.yaml'],
    },
)