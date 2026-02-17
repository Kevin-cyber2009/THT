#!/usr/bin/env python3
# train_unified.py
"""
Script training UNIFIED - Tự động detect Traditional hoặc Hybrid mode
Dựa vào config để quyết định dùng deep learning hay không
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.features import FeatureExtractor
from src.classifier import VideoClassifier
from src.utils import setup_logging, load_config, ensure_dir


def collect_video_paths(data_dir: str):
    """Thu thập đường dẫn videos từ data directory"""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory không tồn tại: {data_dir}")
    
    real_dir = data_path / 'real'
    fake_dir = data_path / 'fake'
    
    if not real_dir.exists() or not fake_dir.exists():
        raise FileNotFoundError(f"Cần có thư mục 'real' và 'fake' trong {data_dir}")
    
    video_paths = []
    labels = []
    
    # Real videos (label = 0)
    for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm']:
        for video in real_dir.glob(ext):
            video_paths.append(str(video))
            labels.append(0)
    
    # Fake videos (label = 1)
    for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm']:
        for video in fake_dir.glob(ext):
            video_paths.append(str(video))
            labels.append(1)
    
    print(f"✓ Tìm thấy {sum(l==0 for l in labels)} real và {sum(l==1 for l in labels)} fake videos")
    
    return video_paths, labels


def extract_features_batch(
    video_paths: list,
    labels: list,
    feature_extractor: FeatureExtractor,
    cache_file: str = None
):
    """
    Trích xuất features từ batch videos
    Tự động sử dụng traditional + deep nếu enabled trong config
    """
    logger = logging.getLogger('hybrid_detector')
    
    # Check cache
    if cache_file and Path(cache_file).exists():
        logger.info(f"Loading features từ cache: {cache_file}")
        cache_data = np.load(cache_file, allow_pickle=True)
        return (
            cache_data['features'],
            cache_data['labels'],
            cache_data['feature_names'].tolist()
        )
    
    features_list = []
    valid_labels = []
    feature_names = None
    
    # Print feature extractor info
    info = feature_extractor.get_feature_info()
    logger.info("=" * 80)
    logger.info("FEATURE EXTRACTION MODE")
    logger.info("=" * 80)
    logger.info(f"Mode: {'HYBRID (Traditional + Deep Learning)' if info['use_deep_learning'] else 'TRADITIONAL ONLY'}")
    logger.info(f"Total features: {info['total_features']}")
    logger.info(f"  - Traditional: {info['traditional_features']}")
    logger.info(f"  - Deep Learning: {info['deep_features']}")
    if info['use_deep_learning'] and 'deep_model' in info:
        logger.info(f"  - Deep Model: {info['deep_model']}")
    logger.info("=" * 80)
    
    logger.info(f"\nExtracting features từ {len(video_paths)} videos...")
    
    for video_path, label in tqdm(zip(video_paths, labels), total=len(video_paths)):
        try:
            # Extract features (tự động dùng traditional + deep nếu enabled)
            features_dict, metadata = feature_extractor.extract_from_video(video_path)
            
            # Get feature names từ video đầu tiên
            if feature_names is None:
                feature_names = feature_extractor.get_feature_names()
                logger.info(f"\n✓ Feature dimension: {len(feature_names)}")
            
            # Convert to vector
            feature_vector = feature_extractor.features_to_vector(
                features_dict,
                feature_names
            )
            
            # Handle NaN/Inf
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
            
            features_list.append(feature_vector)
            valid_labels.append(label)
            
        except Exception as e:
            logger.error(f"Lỗi khi xử lý {video_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(features_list) == 0:
        raise ValueError("Không extract được features nào!")
    
    features_matrix = np.array(features_list)
    labels_array = np.array(valid_labels)
    
    logger.info(f"\n✓ Extracted {len(features_list)} feature vectors, shape: {features_matrix.shape}")
    
    # Cache features
    if cache_file:
        ensure_dir(Path(cache_file).parent)
        np.savez(
            cache_file,
            features=features_matrix,
            labels=labels_array,
            feature_names=np.array(feature_names)
        )
        logger.info(f"✓ Đã cache features tại: {cache_file}")
    
    return features_matrix, labels_array, feature_names


def main():
    parser = argparse.ArgumentParser(
        description='Training Hybrid++ Detector (Unified Traditional + Deep Learning)'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Thư mục chứa data (cần có folder real/ và fake/)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='File cấu hình (config.yaml hoặc config_deep.yaml)'
    )
    
    parser.add_argument(
        '--output_model',
        type=str,
        default=None,
        help='Đường dẫn lưu model (auto-generate nếu không chỉ định)'
    )
    
    parser.add_argument(
        '--cache_features',
        type=str,
        default=None,
        help='File cache features (auto-generate nếu không chỉ định)'
    )
    
    parser.add_argument(
        '--cv_folds',
        type=int,
        default=5,
        help='Số folds cho cross-validation'
    )
    
    parser.add_argument(
        '--skip_cv',
        action='store_true',
        help='Skip cross-validation, chỉ train trên toàn bộ data'
    )
    
    parser.add_argument(
        '--force_cpu',
        action='store_true',
        help='Force CPU (không dùng GPU cho deep learning)'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Force CPU if requested
    if args.force_cpu and 'deep_learning' in config:
        config['deep_learning']['use_gpu'] = False
        logger.info("Forced CPU mode for deep learning")
    
    # Auto-generate output paths
    use_deep = config.get('features', {}).get('use_deep_features', False)
    
    if args.output_model is None:
        if use_deep:
            args.output_model = 'models/hybrid_detector.pkl'
        else:
            args.output_model = 'models/traditional_detector.pkl'
    
    if args.cache_features is None:
        if use_deep:
            args.cache_features = 'cache/features_hybrid.npz'
        else:
            args.cache_features = 'cache/features_traditional.npz'
    
    # Setup logging
    logger = setup_logging(config)
    logger.info("=" * 80)
    logger.info("HYBRID++ DETECTOR - UNIFIED TRAINING")
    logger.info("=" * 80)
    logger.info(f"Config file: {args.config}")
    logger.info(f"Output model: {args.output_model}")
    logger.info(f"Feature cache: {args.cache_features}")
    
    # Collect video paths
    logger.info(f"\nThu thập videos từ: {args.data_dir}")
    video_paths, labels = collect_video_paths(args.data_dir)
    
    # Initialize feature extractor
    logger.info("\nKhởi tạo Feature Extractor...")
    feature_extractor = FeatureExtractor(config)
    
    # Extract features (tự động traditional + deep nếu enabled)
    X, y, feature_names = extract_features_batch(
        video_paths,
        labels,
        feature_extractor,
        cache_file=args.cache_features
    )
    
    # Print dataset statistics
    logger.info("=" * 80)
    logger.info("DATASET STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Tổng số samples: {len(X)}")
    logger.info(f"  - Real videos: {np.sum(y == 0)}")
    logger.info(f"  - Fake videos: {np.sum(y == 1)}")
    logger.info(f"Feature dimension: {X.shape[1]}")
    logger.info(f"Class balance: {np.sum(y == 1) / len(y) * 100:.1f}% fake")
    
    # Initialize classifier
    logger.info("=" * 80)
    logger.info("TRAINING CLASSIFIER")
    logger.info("=" * 80)
    classifier = VideoClassifier(config)
    
    # Cross-validation (optional)
    if not args.skip_cv:
        logger.info(f"Running {args.cv_folds}-fold cross-validation...")
        cv_results = classifier.cross_validate(X, y, cv=args.cv_folds)
        logger.info(f"CV Results:")
        logger.info(f"  Mean AUC: {cv_results['mean_auc']:.4f} ± {cv_results['std_auc']:.4f}")
    
    # Train final model trên toàn bộ data
    logger.info("Training final model trên toàn bộ dataset...")
    metrics = classifier.train(X, y, feature_names=feature_names)
    
    # Print metrics
    logger.info("=" * 80)
    logger.info("TRAINING METRICS")
    logger.info("=" * 80)
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")
    
    # Feature importance
    logger.info("=" * 80)
    logger.info("TOP 15 FEATURE IMPORTANCE")
    logger.info("=" * 80)
    importance = classifier.get_feature_importance()
    if importance:
        sorted_features = sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:15]
        
        for i, (feat_name, imp) in enumerate(sorted_features, 1):
            logger.info(f"{i:2d}. {feat_name:40s}: {imp:.4f}")
    
    # Save model
    logger.info("=" * 80)
    ensure_dir(Path(args.output_model).parent)
    classifier.save(args.output_model)
    
    # Save training summary
    feature_info = feature_extractor.get_feature_info()
    
    summary = {
        'mode': 'hybrid' if use_deep else 'traditional',
        'dataset': {
            'total_samples': int(len(X)),
            'real_samples': int(np.sum(y == 0)),
            'fake_samples': int(np.sum(y == 1)),
            'feature_dimension': int(X.shape[1])
        },
        'features': feature_info,
        'metrics': {k: float(v) for k, v in metrics.items()},
        'model_path': args.output_model,
        'feature_names': feature_names,
        'config_file': args.config
    }
    
    if not args.skip_cv:
        summary['cv_results'] = cv_results
    
    summary_path = Path(args.output_model).parent / 'training_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Training summary đã lưu tại: {summary_path}")
    logger.info("=" * 80)
    logger.info("✓ TRAINING HOÀN TẤT!")
    logger.info("=" * 80)
    
    # Print mode summary
    if use_deep:
        logger.info("\n🚀 Trained in HYBRID mode (Traditional + Deep Learning)")
        logger.info(f"   Deep model: {feature_info.get('deep_model', 'N/A')}")
    else:
        logger.info("\n📊 Trained in TRADITIONAL mode (Forensic + Reality only)")


if __name__ == '__main__':
    main()