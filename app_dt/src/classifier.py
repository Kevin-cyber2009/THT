# src/classifier.py
"""
Module classifier: LightGBM/SVM training và prediction với calibration
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.svm import SVC
import logging

from .utils import load_config, ensure_dir


logger = logging.getLogger('hybrid_detector.classifier')


class VideoClassifier:
    """
    Class training và prediction cho video deepfake detection
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Khởi tạo VideoClassifier
        
        Args:
            config: Dictionary cấu hình
        """
        if config is None:
            config = load_config()
        
        self.config = config
        self.classifier_config = config.get('classifier', {})
        self.model_type = self.classifier_config.get('model_type', 'lightgbm')
        
        # Model và scaler
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
        # Calibrator
        self.calibrator = None
        self.calibration_method = self.classifier_config.get('calibration_method', 'isotonic')
        
        logger.info(f"VideoClassifier initialized, model_type: {self.model_type}")
    
    def _create_model(self):
        """
        Tạo model theo config
        
        Returns:
            Model instance
        """
        if self.model_type == 'lightgbm':
            model = lgb.LGBMClassifier(
                num_leaves=self.classifier_config.get('lgbm_num_leaves', 31),
                max_depth=self.classifier_config.get('lgbm_max_depth', 6),
                learning_rate=self.classifier_config.get('lgbm_learning_rate', 0.05),
                n_estimators=self.classifier_config.get('lgbm_n_estimators', 100),
                min_child_samples=self.classifier_config.get('lgbm_min_child_samples', 20),
                random_state=42,
                verbose=-1
            )
        elif self.model_type == 'svm':
            model = SVC(
                kernel=self.classifier_config.get('svm_kernel', 'rbf'),
                C=self.classifier_config.get('svm_C', 1.0),
                gamma=self.classifier_config.get('svm_gamma', 'scale'),
                probability=True,
                random_state=42
            )
        else:
            raise ValueError(f"Model type không hỗ trợ: {self.model_type}")
        
        return model
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Training model trên data
        
        Args:
            X: Feature matrix (N, D)
            y: Labels (N,) - 0: real, 1: fake
            feature_names: Tên các features
            
        Returns:
            Dictionary training metrics
        """
        logger.info(f"Training model trên {len(X)} samples...")
        
        if feature_names is not None:
            self.feature_names = feature_names
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create và train model
        self.model = self._create_model()
        self.model.fit(X_scaled, y)
        
        # Calibrate probabilities
        cv_folds = self.classifier_config.get('calibration_cv', 5)
        self.calibrator = CalibratedClassifierCV(
            self.model,
            method=self.calibration_method,
            cv=min(cv_folds, len(X) // 2)  # Avoid too many folds for small data
        )
        self.calibrator.fit(X_scaled, y)
        
        # Evaluate
        metrics = self.evaluate(X, y)
        
        logger.info(f"Training hoàn tất - AUC: {metrics.get('auc', 0):.3f}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dự đoán labels và probabilities
        
        Args:
            X: Feature matrix (N, D)
            
        Returns:
            Tuple (predictions, probabilities)
            predictions: 0/1 labels
            probabilities: xác suất class 1 (fake)
        """
        if self.model is None:
            raise ValueError("Model chưa được train. Gọi train() hoặc load() trước.")
        
        X_scaled = self.scaler.transform(X)
        
        # Dùng calibrated probabilities
        if self.calibrator is not None:
            probs = self.calibrator.predict_proba(X_scaled)[:, 1]
        else:
            probs = self.model.predict_proba(X_scaled)[:, 1]
        
        # Threshold tại 0.5
        preds = (probs >= 0.5).astype(int)
        
        return preds, probs
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Đánh giá model với nhiều metrics
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Dictionary metrics
        """
        preds, probs = self.predict(X)
        
        metrics = {}
        
        # AUC
        try:
            metrics['auc'] = float(roc_auc_score(y, probs))
        except:
            metrics['auc'] = 0.0
        
        # Precision/Recall
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        
        metrics['accuracy'] = float(accuracy_score(y, preds))
        metrics['precision'] = float(precision_score(y, preds, zero_division=0))
        metrics['recall'] = float(recall_score(y, preds, zero_division=0))
        metrics['f1'] = float(f1_score(y, preds, zero_division=0))
        
        # FPR at TPR=0.9
        fpr, tpr, thresholds = roc_curve(y, probs)
        target_tpr = 0.9
        idx = np.argmin(np.abs(tpr - target_tpr))
        metrics['fpr_at_tpr_0.9'] = float(fpr[idx])
        
        return metrics
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5
    ) -> Dict[str, Any]:
        """
        Cross-validation
        
        Args:
            X: Feature matrix
            y: Labels
            cv: Số folds
            
        Returns:
            Dictionary CV results
        """
        logger.info(f"Running {cv}-fold cross-validation...")
        
        X_scaled = self.scaler.fit_transform(X)
        model = self._create_model()
        
        # CV scores
        scores = cross_val_score(
            model, X_scaled, y,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        results = {
            'cv_scores': scores.tolist(),
            'mean_auc': float(np.mean(scores)),
            'std_auc': float(np.std(scores))
        }
        
        logger.info(f"CV AUC: {results['mean_auc']:.3f} ± {results['std_auc']:.3f}")
        
        return results
    
    def save(self, model_path: str):
        """
        Lưu model và scaler
        
        Args:
            model_path: Đường dẫn lưu model (.pkl)
        """
        ensure_dir(Path(model_path).parent)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'calibrator': self.calibrator,
            'feature_names': self.feature_names,
            'config': self.classifier_config
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model đã lưu tại: {model_path}")
    
    def load(self, model_path: str):
        """
        Load model từ file
        
        Args:
            model_path: Đường dẫn model file
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file không tồn tại: {model_path}")
        
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.calibrator = model_data.get('calibrator')
        self.feature_names = model_data.get('feature_names')
        
        logger.info(f"Model đã load từ: {model_path}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Lấy feature importance (chỉ cho tree-based models)
        
        Returns:
            Dictionary {feature_name: importance}
        """
        if self.model_type != 'lightgbm':
            logger.warning("Feature importance chỉ khả dụng cho LightGBM")
            return {}
        
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            return {}
        
        importances = self.model.feature_importances_
        
        if self.feature_names is not None:
            return dict(zip(self.feature_names, importances))
        else:
            return {f'feature_{i}': imp for i, imp in enumerate(importances)}