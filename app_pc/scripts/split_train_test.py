# scripts/split_train_test.py

"""
Script split dataset thành train và test
Chỉ cần nếu dataset chưa có sẵn train/test split
"""

import shutil
from pathlib import Path
import random
import argparse


def split_dataset(source_dir, train_dir, test_dir, test_ratio=0.2, seed=42):
    """
    Split dataset thành train và test
    
    Args:
        source_dir: Thư mục chứa all_videos/real và all_videos/fake
        train_dir: Thư mục train output
        test_dir: Thư mục test output
        test_ratio: Tỷ lệ test (0.2 = 20%)
        seed: Random seed
    """
    random.seed(seed)
    
    source_path = Path(source_dir)
    train_path = Path(train_dir)
    test_path = Path(test_dir)
    
    if not source_path.exists():
        print(f"✗ Source directory not found: {source_path}")
        return False
    
    print("="*70)
    print("SPLITTING DATASET")
    print("="*70)
    print(f"Source:     {source_path}")
    print(f"Train:      {train_path}")
    print(f"Test:       {test_path}")
    print(f"Test ratio: {test_ratio*100}%")
    print(f"Seed:       {seed}")
    print()
    
    for category in ['real', 'fake']:
        print(f"\nProcessing {category}...")
        
        category_source = source_path / category
        
        if not category_source.exists():
            print(f"  ✗ Missing: {category_source}")
            continue
        
        # Get all videos
        videos = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            videos.extend(list(category_source.glob(ext)))
        
        if len(videos) == 0:
            print(f"  ✗ No videos found in {category_source}")
            continue
        
        # Shuffle
        random.shuffle(videos)
        
        # Split
        split_idx = int(len(videos) * (1 - test_ratio))
        train_videos = videos[:split_idx]
        test_videos = videos[split_idx:]
        
        print(f"  Total:  {len(videos)}")
        print(f"  Train:  {len(train_videos)} ({len(train_videos)/len(videos)*100:.1f}%)")
        print(f"  Test:   {len(test_videos)} ({len(test_videos)/len(videos)*100:.1f}%)")
        
        # Create dirs
        train_category_dir = train_path / category
        test_category_dir = test_path / category
        
        train_category_dir.mkdir(parents=True, exist_ok=True)
        test_category_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy to train
        print(f"  Copying to train...")
        for video in train_videos:
            shutil.copy(video, train_category_dir / video.name)
        
        # Copy to test
        print(f"  Copying to test...")
        for video in test_videos:
            shutil.copy(video, test_category_dir / video.name)
        
        print(f"  ✓ Done")
    
    print("\n" + "="*70)
    print("✓ Split complete!")
    print("\nNext steps:")
    print(f"  1. Check data: python scripts/check_data.py --data-dir {train_dir}")
    print(f"  2. Train:      python train_classifier.py --data {train_dir}")
    print(f"  3. Evaluate:   python scripts/evaluate_batch.py --test-dir {test_dir}")
    print("="*70)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Split dataset into train and test"
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Source directory (chứa real/ và fake/)'
    )
    parser.add_argument(
        '--train',
        type=str,
        default='data/train',
        help='Train output directory (default: data/train)'
    )
    parser.add_argument(
        '--test',
        type=str,
        default='data/test',
        help='Test output directory (default: data/test)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.2,
        help='Test ratio (default: 0.2 = 20%%)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    split_dataset(
        source_dir=args.source,
        train_dir=args.train,
        test_dir=args.test,
        test_ratio=args.test_ratio,
        seed=args.seed
    )


if __name__ == '__main__':
    main()