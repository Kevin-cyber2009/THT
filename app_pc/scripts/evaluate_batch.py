# scripts/evaluate_batch.py

"""
Script đánh giá model trên toàn bộ test set
Tính các metrics: Accuracy, Precision, Recall, F1, AUC, Confusion Matrix
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import sys

from src.features import FeatureExtractor
from src.classifier import VideoClassifier
from src.utils import load_config, setup_logging


def evaluate_batch(test_dir: Path, model_path: Path, config):
    """
    Đánh giá model trên test set
    
    Args:
        test_dir: Thư mục chứa test/real và test/fake
        model_path: Đường dẫn model
        config: Config dict
        
    Returns:
        metrics: Dictionary metrics
    """
    logger = setup_logging(config)
    
    logger.info("="*70)
    logger.info("BATCH EVALUATION")
    logger.info("="*70)
    
    # Check directories
    if not test_dir.exists():
        logger.error(f"Test directory not found: {test_dir}")
        return None
    
    real_dir = test_dir / 'real'
    fake_dir = test_dir / 'fake'
    
    if not real_dir.exists() or not fake_dir.exists():
        logger.error(f"Missing real/ or fake/ subdirectories in {test_dir}")
        return None
    
    # Load model
    logger.info(f"Loading model: {model_path}")
    classifier = VideoClassifier(config)
    
    try:
        classifier.load(str(model_path))
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None
    
    # Load extractor
    extractor = FeatureExtractor(config)
    feature_names = extractor.get_feature_names()
    
    # Collect all videos
    real_videos = []
    for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
        real_videos.extend(list(real_dir.glob(ext)))
    
    fake_videos = []
    for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
        fake_videos.extend(list(fake_dir.glob(ext)))
    
    logger.info(f"Found {len(real_videos)} real and {len(fake_videos)} fake videos")
    
    if len(real_videos) == 0 or len(fake_videos) == 0:
        logger.error("Not enough videos in test set")
        return None
    
    all_videos = []
    all_labels = []
    
    # Real videos (label = 0)
    for v in real_videos:
        all_videos.append(v)
        all_labels.append(0)
    
    # Fake videos (label = 1)
    for v in fake_videos:
        all_videos.append(v)
        all_labels.append(1)
    
    # Extract features and predict
    predictions = []
    probabilities = []
    failed_videos = []
    
    logger.info("Extracting features and predicting...")
    
    for video_path, true_label in tqdm(
        zip(all_videos, all_labels), 
        total=len(all_videos),
        desc="Processing"
    ):
        try:
            # Extract features
            features, _ = extractor.extract_from_video(str(video_path))
            vector = extractor.features_to_vector(features, feature_names)
            
            # Predict
            pred, prob = classifier.predict(vector.reshape(1, -1))
            
            predictions.append(pred[0])
            probabilities.append(prob[0])
            
        except Exception as e:
            logger.warning(f"Failed to process {video_path.name}: {e}")
            failed_videos.append(video_path.name)
            predictions.append(-1)  # Error flag
            probabilities.append(0.5)
    
    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    all_labels = np.array(all_labels)
    
    # Remove errors
    valid_mask = predictions != -1
    predictions = predictions[valid_mask]
    probabilities = probabilities[valid_mask]
    all_labels = all_labels[valid_mask]
    
    if len(predictions) == 0:
        logger.error("All videos failed to process!")
        return None
    
    # Compute metrics
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, roc_auc_score, confusion_matrix,
        roc_curve
    )
    
    metrics = {
        'accuracy': float(accuracy_score(all_labels, predictions)),
        'precision': float(precision_score(all_labels, predictions, zero_division=0)),
        'recall': float(recall_score(all_labels, predictions, zero_division=0)),
        'f1': float(f1_score(all_labels, predictions, zero_division=0)),
        'auc': float(roc_auc_score(all_labels, probabilities)),
    }
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, predictions)
    
    # FPR at TPR=0.9
    fpr, tpr, thresholds = roc_curve(all_labels, probabilities)
    target_tpr = 0.9
    idx = np.argmin(np.abs(tpr - target_tpr))
    metrics['fpr_at_tpr_0.9'] = float(fpr[idx])
    
    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Test directory:  {test_dir}")
    print(f"Model:           {model_path}")
    print()
    print(f"Total videos:    {len(all_videos)}")
    print(f"Valid:           {len(predictions)}")
    print(f"Failed:          {len(failed_videos)}")
    print()
    print("Metrics:")
    print(f"  Accuracy:      {metrics['accuracy']:.4f}")
    print(f"  Precision:     {metrics['precision']:.4f}")
    print(f"  Recall:        {metrics['recall']:.4f}")
    print(f"  F1 Score:      {metrics['f1']:.4f}")
    print(f"  AUC:           {metrics['auc']:.4f}")
    print(f"  FPR@TPR=0.9:   {metrics['fpr_at_tpr_0.9']:.4f}")
    print()
    print("Confusion Matrix:")
    print(f"                Predicted")
    print(f"              Real  Fake")
    print(f"Actual Real   {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"       Fake   {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    if failed_videos:
        print(f"\nFailed videos ({len(failed_videos)}):")
        for v in failed_videos[:10]:
            print(f"  - {v}")
        if len(failed_videos) > 10:
            print(f"  ... and {len(failed_videos)-10} more")
    
    print("="*70)
    
    # Save results
    output_file = test_dir.parent / 'evaluation_results.json'
    
    results = {
        'test_directory': str(test_dir),
        'model_path': str(model_path),
        'metrics': metrics,
        'confusion_matrix': cm.tolist(),
        'num_videos': len(all_videos),
        'num_valid': len(predictions),
        'num_failed': len(failed_videos),
        'failed_videos': failed_videos
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    logger.info("Evaluation complete")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model on test set"
    )
    parser.add_argument(
        '--test-dir',
        type=str,
        required=True,
        help='Test directory (chứa real/ và fake/)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/detector.pkl',
        help='Model path (default: models/detector.pkl)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Config file (default: config.yaml)'
    )
    
    args = parser.parse_args()
    
    test_dir = Path(args.test_dir)
    model_path = Path(args.model)
    
    if not model_path.exists():
        print(f"✗ Model not found: {model_path}")
        print("\nTrain a model first:")
        print(f"  python train_classifier.py --data data/train --output {model_path}")
        sys.exit(1)
    
    config = load_config(args.config)
    
    metrics = evaluate_batch(test_dir, model_path, config)
    
    if metrics is None:
        sys.exit(1)
    
    # Exit with success
    sys.exit(0)


if __name__ == '__main__':
    main()