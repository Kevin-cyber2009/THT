#!/usr/bin/env python3
# quick_fix_deep.py
"""
Quick fix cho deep_features import error
"""

import os
import shutil
from pathlib import Path

print("=" * 80)
print("QUICK FIX - Deep Features Import Error")
print("=" * 80)

# Check if deep_features.py exists
deep_features_path = Path("src/deep_features.py")

if not deep_features_path.exists():
    print("\n✗ src/deep_features.py NOT FOUND")
    print("\nCopy from outputs:")
    print("  cp outputs/src/deep_features_fixed.py src/deep_features.py")
else:
    print("\n✓ src/deep_features.py exists")
    
    # Check if it's correct
    with open(deep_features_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for key classes
    has_extractor = 'class DeepFeatureExtractor' in content
    has_ensemble = 'class EnsembleDeepExtractor' in content
    
    if has_extractor and has_ensemble:
        print("✓ File seems correct (has both classes)")
        
        # Try to import
        print("\nTesting import...")
        try:
            from src.deep_features import DeepFeatureExtractor
            print("✓ Import successful!")
            
            # Check if PyTorch is installed
            try:
                import torch
                print(f"✓ PyTorch installed: {torch.__version__}")
                
                print("\n" + "=" * 80)
                print("✅ EVERYTHING OK!")
                print("=" * 80)
                print("\nYou can now train with deep learning:")
                print("  python train.py --data_dir data/")
                
            except ImportError:
                print("\n✗ PyTorch NOT installed")
                print("\nInstall PyTorch:")
                print("  pip install torch==2.1.0+cpu torchvision==0.16.0+cpu --index-url https://download.pytorch.org/whl/cpu")
                
        except Exception as e:
            print(f"\n✗ Import failed: {e}")
            print("\nFile may be corrupted. Replace it:")
            print("  cp outputs/src/deep_features_fixed.py src/deep_features.py")
    else:
        print("✗ File incomplete")
        print(f"  Has DeepFeatureExtractor: {has_extractor}")
        print(f"  Has EnsembleDeepExtractor: {has_ensemble}")
        
        # Backup old file
        backup_path = Path("src/deep_features.py.backup")
        shutil.copy(deep_features_path, backup_path)
        print(f"\n✓ Backed up to: {backup_path}")
        
        # Replace with fixed version
        fixed_path = Path("outputs/src/deep_features_fixed.py")
        if fixed_path.exists():
            shutil.copy(fixed_path, deep_features_path)
            print(f"✓ Replaced with fixed version")
            print("\nPlease re-run your training command")
        else:
            print(f"\n✗ Fixed version not found: {fixed_path}")
            print("Please download deep_features_fixed.py and copy to src/")

print("\n" + "=" * 80)