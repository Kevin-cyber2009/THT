# train_classifier.py

"""
Script train classifier trên dataset videos
"""

import argparse
import json
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm

from src.utils import setup_logging, load_config, check_ffmpeg
from src.features import FeatureExtractor
from src.classifier import VideoClassifier


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Video AI Detector Classifier"
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Thư mục chứa videos (real/ và fake/ subfolders)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/detector.pkl',
        help='Đường dẫn lưu model'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='File cấu hình'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Số folds cho cross-validation'
    )
    return parser.parse_args()


def load_dataset(data_dir: Path, extractor: FeatureExtractor, logger):
    """
    Load dataset từ thư mục real/ và fake/
    
    Returns:
        X, y, feature_names
    """
    logger.info("Loading dataset...")
    
    real_dir = data_dir / 'real'
    fake_dir = data_dir / 'fake'
    
    features_list = []
    labels = []
    
    # Load real videos
    if real_dir.exists():
        real_videos = list(real_dir.glob('*.mp4')) + list(real_dir.glob('*.avi'))
        logger.info(f"Tìm thấy {len(real_videos)} real videos")
        
        for video_path in tqdm(real_videos, desc="Processing REAL"):
            try:
                features, _ = extractor.extract_from_video(str(video_path))
                features_list.append(features)
                labels.append(0)  # 0 = real
            except Exception as e:
                logger.warning(f"Lỗi khi xử lý {video_path.name}: {e}")
    
    # Load fake videos
    if fake_dir.exists():
        fake_videos = list(fake_dir.glob('*.mp4')) + list(fake_dir.glob('*.avi'))
        logger.info(f"Tìm thấy {len(fake_videos)} fake videos")
        
        for video_path in tqdm(fake_videos, desc="Processing FAKE"):
            try:
                features, _ = extractor.extract_from_video(str(video_path))
                features_list.append(features)
                labels.append(1)  # 1 = fake
            except Exception as e:
                logger.warning(f"Lỗi khi xử lý {video_path.name}: {e}")
    
    if len(features_list) == 0:
        raise ValueError("Không tìm thấy video nào. Đảm bảo có thư mục real/ và fake/")
    
    # Convert to arrays
    feature_names = extractor.get_feature_names()
    X = np.array([extractor.features_to_vector(f, feature_names) for f in features_list])
    y = np.array(labels)
    
    logger.info(f"Dataset: {len(X)} samples, {X.shape[1]} features")
    logger.info(f"Class distribution: {np.sum(y==0)} real, {np.sum(y==1)} fake")
    
    return X, y, feature_names


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(config)
    logger.info("="*60)
    logger.info("Training Video AI Detector")
    logger.info("="*60)
    
    # Check FFmpeg
    try:
        check_ffmpeg()
    except RuntimeError as e:
        logger.error(f"✗ {e}")
        return
    
    # Check data directory
    data_dir = Path(args.data)
    if not data_dir.exists():
        logger.error(f"✗ Data directory không tồn tại: {data_dir}")
        return
    
    logger.info(f"Data directory: {data_dir}")
    
    # Initialize components
    extractor = FeatureExtractor(config)
    classifier = VideoClassifier(config)
    
    # Load dataset
    try:
        X, y, feature_names = load_dataset(data_dir, extractor, logger)
    except Exception as e:
        logger.error(f"✗ Lỗi khi load dataset: {e}")
        return
    
    # Cross-validation
    logger.info("\n" + "="*60)
    logger.info("Cross-Validation")
    logger.info("="*60)
    
    cv_results = classifier.cross_validate(X, y, cv=args.cv_folds)
    logger.info(f"CV AUC: {cv_results['mean_auc']:.3f} ± {cv_results['std_auc']:.3f}")
    
    # Train final model
    logger.info("\n" + "="*60)
    logger.info("Training Final Model")
    logger.info("="*60)
    
    metrics = classifier.train(X, y, feature_names)
    
    logger.info(f"Training Metrics:")
    logger.info(f"  AUC:       {metrics['auc']:.3f}")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.3f}")
    logger.info(f"  Precision: {metrics['precision']:.3f}")
    logger.info(f"  Recall:    {metrics['recall']:.3f}")
    logger.info(f"  F1:        {metrics['f1']:.3f}")
    
    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    classifier.save(str(output_path))
    logger.info(f"\n✓ Model saved: {output_path}")
    
    # Save metrics
    metrics_path = output_path.parent / f"{output_path.stem}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({
            'cv_results': cv_results,
            'training_metrics': metrics
        }, f, indent=2)
    
    logger.info(f"✓ Metrics saved: {metrics_path}")
    
    # Feature importance
    if config.get('classifier', {}).get('model_type') == 'lightgbm':
        importances = classifier.get_feature_importance()
        logger.info("\nTop 10 Important Features:")
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
        for name, imp in sorted_imp:
            logger.info(f"  {name}: {imp:.4f}")
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING HOÀN TẤT")
    logger.info("="*60)


if __name__ == '__main__':
    main()