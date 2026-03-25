import cv2
from pathlib import Path
import argparse
import sys


def check_video(video_path):
    try:
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return False, "Cannot open"
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        
        if fps == 0:
            return False, "Invalid FPS (0)"
        
        duration = frame_count / fps if fps > 0 else 0
        
        if duration < 2.0:
            return False, f"Too short ({duration:.1f}s, need >= 2s)"
        
        if frame_count < 20:
            return False, f"Too few frames ({frame_count}, need >= 20)"
        
        if width == 0 or height == 0:
            return False, f"Invalid resolution ({width}x{height})"
        
        return True, f"OK ({duration:.1f}s, {frame_count} frames, {width}x{height})"
        
    except Exception as e:
        return False, f"Error: {str(e)}"


def check_dataset(data_dir):
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"✗ Directory not found: {data_path}")
        return False
    
    print("="*70)
    print("CHECKING DATASET QUALITY")
    print("="*70)
    print(f"Directory: {data_path}")
    print()
    
    overall_valid = True
    
    for category in ['real', 'fake']:
        category_dir = data_path / category
        
        if not category_dir.exists():
            print(f"\n✗ Missing directory: {category_dir}")
            overall_valid = False
            continue
        
        # Find all videos
        videos = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            videos.extend(list(category_dir.glob(ext)))
        
        if len(videos) == 0:
            print(f"\n{category.upper()}: ⚠️  No videos found")
            overall_valid = False
            continue
        
        print(f"\n{category.upper()}: {len(videos)} videos")
        print("-" * 70)
        
        valid_count = 0
        issues = []
        
        for video in videos:
            is_valid, msg = check_video(video)
            
            if is_valid:
                valid_count += 1
            else:
                issues.append((video.name, msg))
        
        if valid_count == len(videos):
            print(f"  ✓ All videos valid: {valid_count}/{len(videos)}")
        else:
            print(f"  ⚠️  Valid: {valid_count}/{len(videos)}")
            print(f"  ✗ Issues: {len(issues)}")
            
            if issues:
                print(f"\n  Problematic videos:")
                for video_name, msg in issues[:15]:
                    print(f"    - {video_name}: {msg}")
                
                if len(issues) > 15:
                    print(f"    ... and {len(issues)-15} more")
        
        min_required = 50
        if valid_count < min_required:
            print(f"\n  ⚠️  WARNING: Only {valid_count} valid videos (recommended >= {min_required})")
            overall_valid = False
    
    print("\n" + "="*70)
    
    if overall_valid:
        print("✓ Dataset looks good!")
        print("\nNext step:")
        print("  python train_classifier.py --data", data_dir)
        return True
    else:
        print("⚠️  Dataset has issues. Please fix them before training.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Check dataset quality for training"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Data directory (chứa real/ và fake/)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed info for all videos'
    )
    
    args = parser.parse_args()
    
    success = check_dataset(args.data_dir)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()