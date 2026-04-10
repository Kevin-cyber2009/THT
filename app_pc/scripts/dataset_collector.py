"""
Dataset Collection Guide for AI Video Detection
==============================================

Public Datasets Available:

1. ScaleDF (Largest)
   - URL: https://huggingface.co/datasets/WenhaoWang/ScaleDF
   - Size: 5.8+ million images/videos
   - Content: Comprehensive deepfake detection dataset
   - License: CC BY-NC

2. AV-Deepfake1M
   - URL: https://huggingface.co/datasets/ControlNet/AV-Deepfake1M
   - Size: 1M+ videos
   - Content: Audio-visual deepfakes
   - License: CC BY-NC 4.0

3. OpenFake
   - URL: https://huggingface.co/datasets/ComplexDataLab/OpenFake
   - URL: https://arxiv.org/html/2509.09495v2
   - Content: Real-world deepfake detection

4. Deepfake-Eval-2024
   - URL: https://huggingface.co/datasets/nuriachandra/Deepfake-Eval-2024
   - Content: 2024 deepfakes benchmark

5. Celeb-DF++
   - URL: https://github.com/OUC-VAS/Celeb-DF-PP
   - Size: ~1000+ videos
   - Content: High-quality face swaps

6. FaceForensics++
   - URL: https://www.kaggle.com/datasets/xdxd003/ff-c23
   - URL: https://www.kaggle.com/datasets/hungle3401/faceforensics
   - Content: 1000+ videos with multiple compression levels

7. DFDC Preview
   - URL: https://huggingface.co/papers/1910.08854
   - Size: 5K+ videos
   - Content: Facebook's Deepfake Detection Challenge

8. Mendeley Deepfake Dataset
   - URL: https://data.mendeley.com/datasets/pdcp9mjy3z/1
   - Content: Roop and Akool AI technologies

9. GenVidBench
   - URL: https://genvidbench.github.io/
   - Size: 6 Million benchmark
   - Content: AI-generated video detection

10. ExDDV
    - URL: https://arxiv.org/html/2503.14421v1
    - Content: Explainable deepfake detection

Usage:
------
1. Install huggingface_hub: pip install huggingface_hub
2. Download datasets using scripts in this folder
"""

import os
import subprocess
from pathlib import Path

DATASETS = {
    "scaledf": {
        "name": "ScaleDF",
        "url": "https://huggingface.co/datasets/WenhaoWang/ScaleDF",
        "description": "Largest deepfake dataset - 5.8M+ samples",
    },
    "openfake": {
        "name": "OpenFake",
        "url": "https://huggingface.co/datasets/ComplexDataLab/OpenFake",
        "description": "Real-world deepfake detection",
    },
    "deepfake_eval_2024": {
        "name": "Deepfake-Eval-2024",
        "url": "https://huggingface.co/datasets/nuriachandra/Deepfake-Eval-2024",
        "description": "2024 deepfakes benchmark",
    },
    "ff_images": {
        "name": "FaceForensics++ Images",
        "url": "https://huggingface.co/datasets/nikhilny25/ff-images-dataset",
        "description": "Preprocessed FaceForensics++ images",
    },
}


def download_huggingface_dataset(dataset_name: str, output_dir: str = "datasets"):
    """Download dataset from Hugging Face."""
    cmd = [
        "python", "-m", "huggingface_hub", "download",
        dataset_name,
        "--repo-type", "dataset",
        "--local-dir", output_dir,
    ]
    
    print(f"Downloading {dataset_name} to {output_dir}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✓ Downloaded {dataset_name}")
    else:
        print(f"✗ Failed: {result.stderr}")


def main():
    print("=" * 60)
    print("AI Video Detection Dataset Collection")
    print("=" * 60)
    
    output_dir = Path("datasets")
    output_dir.mkdir(exist_ok=True)
    
    print("\nAvailable datasets:")
    for key, info in DATASETS.items():
        print(f"  [{key}] {info['name']}")
        print(f"       {info['description']}")
    
    print("\n" + "=" * 60)
    print("To download a dataset, run:")
    print("  python dataset_collector.py --download scaledf")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Dataset Collection")
    parser.add_argument("--download", choices=list(DATASETS.keys()), help="Dataset to download")
    parser.add_argument("--output", default="datasets", help="Output directory")
    
    args = parser.parse_args()
    
    if args.download:
        download_huggingface_dataset(args.download, args.output)
    else:
        main()
