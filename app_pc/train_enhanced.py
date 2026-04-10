"""
Enhanced Training Script for AI Video Detection
=============================================

Trains with:
- Traditional forensic features (17)
- Reality compliance features (11)
- Deep learning features (11)
- Face analysis features (~30) [NEW]
- Temporal features (~25) [NEW]

Total: ~94 features
"""

import argparse
import json
import logging
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

from src.utils import setup_logging, load_config, check_ffmpeg, ensure_dir
from src.features import FeatureExtractor
from src.forensic import ForensicAnalyzer
from src.reality_engine import RealityEngine
from src.preprocessing import VideoPreprocessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Enhanced Training Video AI Detector v2.0"
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
        default='models/hybrid_detector_v2.pkl',
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
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Tỷ lệ test set'
    )
    parser.add_argument(
        '--skip-deep',
        action='store_true',
        help='Bỏ qua deep learning features'
    )
    parser.add_argument(
        '--skip-face',
        action='store_true',
        help='Bỏ qua face analysis features'
    )
    parser.add_argument(
        '--skip-temporal',
        action='store_true',
        help='Bỏ qua temporal features'
    )
    return parser.parse_args()


class EnhancedFeatureExtractor:
    """Enhanced feature extractor with face and temporal analysis."""
    
    def __init__(self, config, skip_face=False, skip_temporal=False, skip_deep=False):
        self.config = config
        self.skip_face = skip_face
        self.skip_temporal = skip_temporal
        self.skip_deep = skip_deep
        
        self.preprocessor = VideoPreprocessor(config)
        self.forensic = ForensicAnalyzer(config)
        self.reality = RealityEngine(config)
        
        self.face_analyzer = None
        self.temporal_analyzer = None
        self.deep_extractor = None
        
        if not skip_face:
            try:
                from src.face_analyzer import FaceAnalyzer
                self.face_analyzer = FaceAnalyzer(config)
                logging.info("FaceAnalyzer initialized")
            except ImportError:
                logging.warning("FaceAnalyzer not available")
        
        if not skip_temporal:
            try:
                from src.temporal_features import TemporalAnalyzer
                self.temporal_analyzer = TemporalAnalyzer(config)
                logging.info("TemporalAnalyzer initialized")
            except ImportError:
                logging.warning("TemporalAnalyzer not available")
        
        if not skip_deep:
            try:
                prefer_onnx = config.get('deep_learning', {}).get('prefer_onnx', False)
                if prefer_onnx:
                    from src.deep_features_onnx import DeepFeatureExtractor, EnsembleDeepExtractor
                else:
                    from src.deep_features import DeepFeatureExtractor, EnsembleDeepExtractor
                
                use_ensemble = config.get('deep_learning', {}).get('use_ensemble', False)
                if use_ensemble:
                    self.deep_extractor = EnsembleDeepExtractor(config)
                else:
                    self.deep_extractor = DeepFeatureExtractor(config)
                logging.info("Deep extractor initialized")
            except Exception as e:
                logging.warning(f"Deep extractor not available: {e}")
    
    def extract_features(self, frames):
        """Extract all features from frames."""
        all_features = {}
        
        forensic_features = self.forensic.analyze(frames)
        all_features.update(forensic_features)
        
        reality_features = self.reality.analyze(frames)
        all_features.update(reality_features)
        
        if self.face_analyzer is not None:
            try:
                face_features = self.face_analyzer.extract_all_features(
                    frames.tolist() if hasattr(frames, 'tolist') else frames
                )
                for key, value in face_features.items():
                    if key not in ['frames_analyzed']:
                        all_features[f'face_{key}'] = value
                logging.debug(f"Extracted {len(face_features)} face features")
            except Exception as e:
                logging.warning(f"Face analysis failed: {e}")
        
        if self.temporal_analyzer is not None:
            try:
                temporal_features = self.temporal_analyzer.extract_all_features(frames)
                for key, value in temporal_features.items():
                    all_features[f'temporal_{key}'] = value
                logging.debug(f"Extracted {len(temporal_features)} temporal features")
            except Exception as e:
                logging.warning(f"Temporal analysis failed: {e}")
        
        if self.deep_extractor is not None:
            try:
                sample_frames = self.config.get('deep_learning', {}).get('sample_frames', 10)
                deep_features = self.deep_extractor.extract_video_features(frames, sample_frames)
                all_features.update(deep_features)
                logging.debug(f"Extracted {len(deep_features)} deep features")
            except Exception as e:
                logging.warning(f"Deep analysis failed: {e}")
        
        return all_features
    
    def extract_from_video(self, video_path):
        """Extract features from video file."""
        frames, metadata = self.preprocessor.preprocess(video_path)
        if len(frames) < 10:
            frames = self.preprocessor.handle_short_video(frames, min_frames=10)
        
        features = self.extract_features(frames)
        return features, metadata
    
    def get_feature_names(self):
        """Get list of feature names."""
        names = [
            'fft_mean', 'fft_std', 'fft_max', 'fft_high_freq_energy', 'fft_radial_slope',
            'dct_mean', 'dct_std', 'dct_dc_mean', 'dct_ac_energy',
            'prnu_mean', 'prnu_std', 'prnu_autocorr', 'prnu_temporal_consistency',
            'flow_mean_magnitude', 'flow_std_magnitude', 'flow_smoothness', 'flow_temporal_consistency',
            'entropy_mean', 'entropy_std', 'entropy_slope',
            'fractal_dim_mean', 'fractal_dim_std',
            'causal_prediction_error', 'causal_predictability',
            'compression_mean', 'compression_std', 'compression_delta_mean', 'complexity_mean',
        ]
        
        if self.deep_extractor is not None:
            names.extend([
                'deep_feat_mean', 'deep_feat_std', 'deep_feat_max', 'deep_feat_min',
                'deep_temporal_var_mean', 'deep_temporal_var_std',
                'deep_l2_norm_mean', 'deep_l2_norm_std',
                'deep_similarity_mean', 'deep_similarity_std', 'deep_sparsity',
            ])
        
        return names
    
    def features_to_vector(self, features, feature_names=None):
        """Convert features dict to numpy array."""
        if feature_names is None:
            feature_names = sorted(features.keys())
        
        vector = []
        for name in feature_names:
            value = features.get(name, 0.0)
            if np.isnan(value) or np.isinf(value):
                value = 0.0
            vector.append(value)
        
        return np.array(vector, dtype=np.float32)


def load_dataset(data_dir, extractor, logger, max_samples=None):
    """Load dataset from directory."""
    logger.info("Loading dataset...")
    
    real_dir = data_dir / 'real'
    fake_dir = data_dir / 'fake'
    
    features_list = []
    labels = []
    video_paths = []
    
    if real_dir.exists():
        real_videos = list(real_dir.glob('*.mp4')) + list(real_dir.glob('*.avi'))
        logger.info(f"Found {len(real_videos)} real videos")
        
        for video_path in tqdm(real_videos, desc="Processing REAL"):
            if max_samples and len(features_list) >= max_samples:
                break
            try:
                features, _ = extractor.extract_from_video(str(video_path))
                features_list.append(features)
                labels.append(0)
                video_paths.append(str(video_path))
            except Exception as e:
                logger.warning(f"Error processing {video_path.name}: {e}")
    
    if fake_dir.exists():
        fake_videos = list(fake_dir.glob('*.mp4')) + list(fake_dir.glob('*.avi'))
        logger.info(f"Found {len(fake_videos)} fake videos")
        
        for video_path in tqdm(fake_videos, desc="Processing FAKE"):
            if max_samples and len(features_list) >= max_samples:
                break
            try:
                features, _ = extractor.extract_from_video(str(video_path))
                features_list.append(features)
                labels.append(1)
                video_paths.append(str(video_path))
            except Exception as e:
                logger.warning(f"Error processing {video_path.name}: {e}")
    
    if len(features_list) == 0:
        raise ValueError("No videos found. Ensure real/ and fake/ subfolders exist.")
    
    feature_names = extractor.get_feature_names()
    X = np.array([extractor.features_to_vector(f, feature_names) for f in features_list])
    y = np.array(labels)
    
    logger.info(f"Dataset: {len(X)} samples, {X.shape[1]} features")
    logger.info(f"Class distribution: {np.sum(y==0)} real, {np.sum(y==1)} fake")
    
    return X, y, feature_names, video_paths


def train_model(X_train, y_train, X_val, y_val, config):
    """Train LightGBM model."""
    clf_config = config.get('classifier', {})
    
    model = lgb.LGBMClassifier(
        num_leaves=clf_config.get('lgbm_num_leaves', 63),
        max_depth=clf_config.get('lgbm_max_depth', 8),
        learning_rate=clf_config.get('lgbm_learning_rate', 0.03),
        n_estimators=clf_config.get('lgbm_n_estimators', 500),
        min_child_samples=clf_config.get('lgbm_min_child_samples', 15),
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    return model


def evaluate_model(model, X, y):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'auc': roc_auc_score(y, y_prob),
    }
    
    return metrics, y_prob


def cross_validate(X, y, config, n_folds=5):
    """Perform cross-validation."""
    logger = logging.getLogger('hybrid_detector')
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    cv_scores = []
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"Fold {fold + 1}/{n_folds}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        model = train_model(X_train_scaled, y_train, X_val_scaled, y_val, config)
        
        metrics, _ = evaluate_model(model, X_val_scaled, y_val)
        cv_scores.append(metrics['auc'])
        fold_metrics.append(metrics)
        
        logger.info(f"  AUC: {metrics['auc']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
    
    logger.info(f"\nCV Results: AUC = {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    return {
        'cv_scores': cv_scores,
        'mean_auc': np.mean(cv_scores),
        'std_auc': np.std(cv_scores),
        'fold_metrics': fold_metrics,
    }


def train_final_model(X, y, config):
    """Train final model on all data."""
    logger = logging.getLogger('hybrid_detector')
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf_config = config.get('classifier', {})
    
    model = lgb.LGBMClassifier(
        num_leaves=clf_config.get('lgbm_num_leaves', 63),
        max_depth=clf_config.get('lgbm_max_depth', 8),
        learning_rate=clf_config.get('lgbm_learning_rate', 0.03),
        n_estimators=200,
        min_child_samples=clf_config.get('lgbm_min_child_samples', 15),
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_scaled, y)
    
    metrics, _ = evaluate_model(model, X_scaled, y)
    
    return model, scaler, metrics


def main():
    args = parse_args()
    
    config = load_config(args.config)
    config['features']['use_deep_features'] = not args.skip_deep
    
    logger = setup_logging(config)
    logger.info("=" * 60)
    logger.info("Enhanced Training Video AI Detector v2.0")
    logger.info("=" * 60)
    logger.info(f"Skip Deep: {args.skip_deep}")
    logger.info(f"Skip Face: {args.skip_face}")
    logger.info(f"Skip Temporal: {args.skip_temporal}")
    
    try:
        check_ffmpeg()
    except RuntimeError as e:
        logger.error(f"FFmpeg error: {e}")
        return
    
    data_dir = Path(args.data)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return
    
    extractor = EnhancedFeatureExtractor(
        config,
        skip_face=args.skip_face,
        skip_temporal=args.skip_temporal,
        skip_deep=args.skip_deep
    )
    
    try:
        X, y, feature_names, video_paths = load_dataset(data_dir, extractor, logger)
    except Exception as e:
        logger.error(f"Dataset load error: {e}")
        return
    
    logger.info(f"\nFeature breakdown:")
    logger.info(f"  - Traditional: 28")
    if not args.skip_deep:
        logger.info(f"  - Deep: 11")
    if not args.skip_face:
        logger.info(f"  - Face: ~30")
    if not args.skip_temporal:
        logger.info(f"  - Temporal: ~25")
    logger.info(f"  - Total: {len(feature_names)}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Cross-Validation")
    logger.info("=" * 60)
    
    cv_results = cross_validate(X, y, config, n_folds=args.cv_folds)
    
    logger.info("\n" + "=" * 60)
    logger.info("Training Final Model")
    logger.info("=" * 60)
    
    model, scaler, train_metrics = train_final_model(X, y, config)
    
    logger.info(f"Training Metrics:")
    logger.info(f"  AUC:       {train_metrics['auc']:.4f}")
    logger.info(f"  Accuracy:  {train_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {train_metrics['precision']:.4f}")
    logger.info(f"  Recall:    {train_metrics['recall']:.4f}")
    logger.info(f"  F1:        {train_metrics['f1']:.4f}")
    
    importances = dict(zip(feature_names, model.feature_importances_))
    logger.info("\nTop 15 Important Features:")
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:15]
    for name, imp in sorted_imp:
        logger.info(f"  {name}: {imp}")
    
    output_path = Path(args.output)
    ensure_dir(str(output_path.parent))
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'config': config,
        'n_features': len(feature_names),
        'cv_results': cv_results,
        'train_metrics': train_metrics,
        'feature_importances': importances,
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    logger.info(f"\nModel saved: {output_path}")
    
    scaler_params_path = output_path.parent / f"{output_path.stem}_scaler_params.json"
    scaler_params = {
        'n_features': scaler.n_features_in_,
        'mean_': scaler.mean_.tolist(),
        'scale_': scaler.scale_.tolist(),
        'feature_names': feature_names,
    }
    with open(scaler_params_path, 'w') as f:
        json.dump(scaler_params, f, indent=2)
    logger.info(f"Scaler params saved: {scaler_params_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"\nTo use this model:")
    logger.info(f"1. Convert to ONNX: python convert_to_onnx.py --model {output_path}")
    logger.info(f"2. Copy ONNX to Android: app_dt/app/src/main/assets/models/")
    logger.info(f"3. Copy scaler params: {scaler_params_path} -> Android assets")


if __name__ == '__main__':
    main()
