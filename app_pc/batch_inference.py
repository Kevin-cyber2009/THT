#!/usr/bin/env python3
# batch_inference.py
"""
Script inference cho nhiều videos cùng lúc
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json
from datetime import datetime

from src.features import FeatureExtractor
from src.classifier import VideoClassifier
from src.fusion import ScoreFusion
from src.stress_lab import StressLab
from src.utils import setup_logging, load_config
from inference import predict_single_video


def collect_videos(input_path: str):
    """
    Thu thập tất cả videos từ thư mục hoặc file list
    
    Args:
        input_path: Thư mục hoặc file txt chứa danh sách videos
        
    Returns:
        List đường dẫn videos
    """
    path = Path(input_path)
    
    if path.is_file() and path.suffix == '.txt':
        # Đọc từ file txt
        with open(path, 'r') as f:
            videos = [line.strip() for line in f if line.strip()]
        return videos
    
    elif path.is_dir():
        # Scan thư mục
        extensions = ['.mp4', '.avi', '.mov', '.mkv']
        videos = []
        
        for ext in extensions:
            videos.extend(list(path.glob(f'*{ext}')))
            videos.extend(list(path.glob(f'**/*{ext}')))  # Recursive
        
        return [str(v) for v in set(videos)]
    
    else:
        raise ValueError(f"Input path không hợp lệ: {input_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Batch Inference cho nhiều videos'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Thư mục chứa videos hoặc file .txt danh sách videos'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='models/hybrid_detector.pkl',
        help='Đường dẫn model đã train'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='File cấu hình'
    )
    
    parser.add_argument(
        '--output_csv',
        type=str,
        default='output/batch_results.csv',
        help='File CSV lưu kết quả'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output/batch_json',
        help='Thư mục lưu JSON từng video'
    )
    
    parser.add_argument(
        '--stress_test',
        action='store_true',
        help='Chạy stress tests (chậm hơn)'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    logger = setup_logging(config)
    
    logger.info("=" * 80)
    logger.info("BATCH INFERENCE")
    logger.info("=" * 80)
    
    # Check model
    if not Path(args.model).exists():
        print(f"❌ Model không tồn tại: {args.model}")
        return
    
    # Collect videos
    logger.info(f"Thu thập videos từ: {args.input}")
    videos = collect_videos(args.input)
    logger.info(f"✓ Tìm thấy {len(videos)} videos")
    
    if len(videos) == 0:
        print("❌ Không tìm thấy video nào")
        return
    
    # Create output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info("Loading model...")
    classifier = VideoClassifier(config)
    classifier.load(args.model)
    
    # Initialize components
    feature_extractor = FeatureExtractor(config)
    fusion_engine = ScoreFusion(config)
    stress_lab = StressLab(config) if args.stress_test else None
    
    # Process videos
    results = []
    failed = []
    
    logger.info(f"Processing {len(videos)} videos...")
    print(f"\n{'='*80}")
    print(f"Processing {len(videos)} videos...")
    print(f"{'='*80}\n")
    
    for video_path in tqdm(videos, desc="Processing"):
        try:
            # Predict
            result = predict_single_video(
                video_path,
                classifier,
                feature_extractor,
                fusion_engine=fusion_engine,
                stress_lab=stress_lab,
                run_stress_tests=args.stress_test
            )
            
            # Save JSON
            video_name = Path(video_path).stem
            json_path = output_dir / f"{video_name}.json"
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            
            # Collect for CSV
            row = {
                'video_name': Path(video_path).name,
                'video_path': video_path,
                'prediction': result['prediction'],
                'probability_fake': result['probability_fake'],
                'num_frames': result['metadata'].get('num_frames', 'N/A'),
                'duration': result['metadata'].get('duration', 'N/A'),
            }
            
            if 'fusion_result' in result:
                fusion = result['fusion_result']
                row['artifact_score'] = result.get('artifact_score', 'N/A')
                row['reality_score'] = result.get('reality_score', 'N/A')
                row['stress_score'] = result.get('stress_score', 'N/A')
                row['final_probability'] = fusion['final_probability']
                row['confidence'] = fusion['confidence']
            
            results.append(row)
            
        except Exception as e:
            logger.error(f"Lỗi xử lý {video_path}: {e}")
            failed.append({
                'video_path': video_path,
                'error': str(e)
            })
    
    # Save CSV
    if results:
        df = pd.DataFrame(results)
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False, encoding='utf-8')
        logger.info(f"✓ Kết quả CSV đã lưu tại: {args.output_csv}")
        
        # Print summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"Total videos: {len(videos)}")
        print(f"Processed: {len(results)}")
        print(f"Failed: {len(failed)}")
        
        if len(results) > 0:
            fake_count = sum(1 for r in results if r['prediction'] == 'FAKE')
            real_count = len(results) - fake_count
            print(f"\nPredictions:")
            print(f"  - REAL: {real_count} ({real_count/len(results)*100:.1f}%)")
            print(f"  - FAKE: {fake_count} ({fake_count/len(results)*100:.1f}%)")
            
            if 'final_probability' in results[0]:
                avg_prob = sum(r['final_probability'] for r in results) / len(results)
                print(f"\nAverage fake probability: {avg_prob:.3f}")
        
        print(f"\nOutput files:")
        print(f"  - CSV: {args.output_csv}")
        print(f"  - JSON dir: {args.output_dir}")
        print(f"{'='*80}\n")
    
    # Save failed list
    if failed:
        failed_file = Path(args.output_csv).parent / 'failed_videos.json'
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump(failed, f, indent=2, ensure_ascii=False)
        logger.warning(f"⚠ Danh sách videos lỗi: {failed_file}")


if __name__ == '__main__':
    main()