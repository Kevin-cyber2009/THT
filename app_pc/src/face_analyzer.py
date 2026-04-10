"""
Face Analyzer Module for AI Video Detection
============================================

Extracts face-specific features that are highly effective for detecting AI-generated videos.

Features extracted:
- Eye blink rate and patterns
- Face symmetry analysis
- Skin texture analysis
- Facial landmark consistency
- Lip sync analysis (if audio available)
- Face deformation metrics
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from scipy.spatial.distance import euclidean
from scipy.signal import find_peaks

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None

logger = logging.getLogger('hybrid_detector.face_analyzer')


@dataclass
class FaceDetectionResult:
    face_box: Tuple[int, int, int, int]
    landmarks: np.ndarray
    confidence: float


class FaceAnalyzer:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        
        self.face_config = self.config.get('face_analysis', {})
        self.min_face_size = self.face_config.get('min_face_size', 64)
        self.blink_history_size = self.face_config.get('blink_history_size', 30)
        self.eye_aspect_ratio_threshold = self.face_config.get('eye_aspect_ratio_threshold', 0.2)
        
        self._mp_face_mesh = None
        self._mp_drawing = None
        self._mp_drawing_styles = None
        
        if MEDIAPIPE_AVAILABLE:
            self._init_mediapipe()
        
        self.blink_history: List[float] = []
        self.landmark_history: List[np.ndarray] = []
        
        logger.info(f"FaceAnalyzer initialized, mediapipe={'available' if MEDIAPIPE_AVAILABLE else 'NOT AVAILABLE'}")

    def _init_mediapipe(self):
        """Initialize MediaPipe face mesh."""
        try:
            self._mp_face_mesh = mp.face_mesh
            self._mp_drawing = mp.drawing_utils
            self._mp_drawing_styles = mp.drawing_styles
            self._face_mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("MediaPipe face mesh initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize MediaPipe: {e}")
            self._mp_face_mesh = None

    def detect_face(self, frame: np.ndarray) -> Optional[FaceDetectionResult]:
        """Detect face in a single frame."""
        if self._mp_face_mesh is None:
            return self._detect_face_haar(frame)
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            h, w = frame.shape[:2]
            points = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in landmarks.landmark])
            
            x_min, y_min = points[:, :2].min(axis=0).astype(int)
            x_max, y_max = points[:, :2].max(axis=0).astype(int)
            
            return FaceDetectionResult(
                face_box=(x_min, y_min, x_max - x_min, y_max - y_min),
                landmarks=points,
                confidence=1.0
            )
        
        return None

    def _detect_face_haar(self, frame: np.ndarray) -> Optional[FaceDetectionResult]:
        """Fallback face detection using Haar cascades."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        ).detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            return FaceDetectionResult(
                face_box=(x, y, w, h),
                landmarks=np.array([]),
                confidence=0.8
            )
        return None

    def extract_eye_features(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Extract eye-related features."""
        features = {}
        
        if len(landmarks) < 468:
            return self._default_eye_features()
        
        left_eye = landmarks[362:1337:5]
        right_eye = landmarks[145:159:5]
        
        def eye_aspect_ratio(eye_points):
            if len(eye_points) < 6:
                return 0.0
            v1 = euclidean(eye_points[1], eye_points[5])
            v2 = euclidean(eye_points[2], eye_points[4])
            h = euclidean(eye_points[0], eye_points[3])
            if h == 0:
                return 0.0
            return (v1 + v2) / (2.0 * h)
        
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        features['eye_aspect_ratio_mean'] = float(np.mean([left_ear, right_ear]))
        features['eye_aspect_ratio_std'] = float(np.std([left_ear, right_ear]))
        
        features['eye_distance_ratio'] = self._calculate_eye_distance_ratio(landmarks)
        
        return features

    def _calculate_eye_distance_ratio(self, landmarks: np.ndarray) -> float:
        """Calculate ratio between eye distance and face width."""
        if len(landmarks) < 468:
            return 0.0
        
        left_eye_center = landmarks[33][:2]
        right_eye_center = landmarks[263][:2]
        eye_distance = euclidean(left_eye_center, right_eye_center)
        
        face_left = landmarks[234][:2]
        face_right = landmarks[454][:2]
        face_width = euclidean(face_left, face_right)
        
        if face_width == 0:
            return 0.0
        return float(eye_distance / face_width)

    def _default_eye_features(self) -> Dict[str, float]:
        """Return default values when face not detected."""
        return {
            'eye_aspect_ratio_mean': 0.25,
            'eye_aspect_ratio_std': 0.05,
            'eye_distance_ratio': 0.3,
        }

    def extract_symmetry_features(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Extract face symmetry features."""
        features = {}
        
        if len(landmarks) < 468:
            return self._default_symmetry_features()
        
        left_indices = [234, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        right_indices = [454, 323, 454, 356, 291, 162, 284, 332, 297, 337]
        
        symmetry_scores = []
        for li, ri in zip(left_indices, right_indices):
            left = landmarks[li][:2]
            right = landmarks[ri][:2]
            center = (landmarks[168][:2] + landmarks[4][:2]) / 2
            
            dist_left = euclidean(left, center)
            dist_right = euclidean(right, center)
            
            if dist_left > 0 and dist_right > 0:
                ratio = min(dist_left, dist_right) / max(dist_left, dist_right)
                symmetry_scores.append(ratio)
        
        features['face_symmetry_score'] = float(np.mean(symmetry_scores)) if symmetry_scores else 1.0
        features['face_symmetry_std'] = float(np.std(symmetry_scores)) if symmetry_scores else 0.0
        
        return features

    def _default_symmetry_features(self) -> Dict[str, float]:
        return {
            'face_symmetry_score': 1.0,
            'face_symmetry_std': 0.0,
        }

    def extract_skin_texture_features(self, frame: np.ndarray, face_box: Tuple[int, int, int, int]) -> Dict[str, float]:
        """Extract skin texture analysis features."""
        x, y, w, h = face_box
        
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        
        if w < 20 or h < 20:
            return self._default_skin_features()
        
        face_roi = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        features = {}
        features['skin_texture_variance'] = float(np.var(gray))
        features['skin_gradient_mean'] = float(np.mean(gradient_magnitude))
        features['skin_gradient_std'] = float(np.std(gradient_magnitude))
        
        edges = cv2.Canny(gray, 50, 150)
        features['skin_edge_density'] = float(np.sum(edges > 0) / edges.size)
        
        return features

    def _default_skin_features(self) -> Dict[str, float]:
        return {
            'skin_texture_variance': 500.0,
            'skin_gradient_mean': 10.0,
            'skin_gradient_std': 5.0,
            'skin_edge_density': 0.1,
        }

    def track_blink(self, landmarks: np.ndarray) -> Optional[float]:
        """Track eye blink state and return current EAR."""
        if len(landmarks) < 468:
            return None
        
        left_eye = landmarks[362:382]
        right_eye = landmarks[1337:1357]
        
        def ear(eye):
            v1 = euclidean(eye[1], eye[5])
            v2 = euclidean(eye[2], eye[4])
            h = euclidean(eye[0], eye[3])
            if h == 0:
                return 0.0
            return (v1 + v2) / (2.0 * h)
        
        left_ear = ear(left_eye)
        right_ear = ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2
        
        self.blink_history.append(avg_ear)
        if len(self.blink_history) > self.blink_history_size:
            self.blink_history.pop(0)
        
        return avg_ear

    def detect_blinks(self) -> Dict[str, float]:
        """Detect blinks from history and return statistics."""
        features = {}
        
        if len(self.blink_history) < 10:
            return self._default_blink_features()
        
        history = np.array(self.blink_history)
        
        threshold = np.mean(history) * 0.7
        below_threshold = history < threshold
        
        blink_starts = []
        in_blink = False
        for i, val in enumerate(below_threshold):
            if val and not in_blink:
                blink_starts.append(i)
                in_blink = True
            elif not val:
                in_blink = False
        
        features['blink_rate'] = float(len(blink_starts) / (len(history) / 30))
        features['blink_rate_normalized'] = float(features['blink_rate'] / 20.0)
        
        if len(self.blink_history) > 0:
            features['avg_eye_openness'] = float(np.mean(self.blink_history))
            features['eye_openness_std'] = float(np.std(self.blink_history))
        else:
            features['avg_eye_openness'] = 0.25
            features['eye_openness_std'] = 0.05
        
        return features

    def _default_blink_features(self) -> Dict[str, float]:
        return {
            'blink_rate': 15.0,
            'blink_rate_normalized': 0.75,
            'avg_eye_openness': 0.25,
            'eye_openness_std': 0.05,
        }

    def analyze_landmark_consistency(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Analyze temporal consistency of facial landmarks."""
        features = {}
        
        self.landmark_history.append(landmarks.copy())
        if len(self.landmark_history) > 30:
            self.landmark_history.pop(0)
        
        if len(self.landmark_history) < 2:
            return self._default_landmark_features()
        
        history_array = np.array(self.landmark_history)
        
        temporal_variance = np.var(history_array, axis=0)
        features['landmark_temporal_variance'] = float(np.mean(temporal_variance))
        features['landmark_temporal_variance_max'] = float(np.max(temporal_variance))
        
        movements = []
        for i in range(1, len(history_array)):
            diff = history_array[i] - history_array[i-1]
            movements.append(np.mean(np.sqrt(np.sum(diff**2, axis=1))))
        
        features['avg_landmark_movement'] = float(np.mean(movements))
        features['landmark_movement_std'] = float(np.std(movements))
        
        return features

    def _default_landmark_features(self) -> Dict[str, float]:
        return {
            'landmark_temporal_variance': 0.0,
            'landmark_temporal_variance_max': 0.0,
            'avg_landmark_movement': 0.0,
            'landmark_movement_std': 0.0,
        }

    def extract_face_deformation_features(self, landmarks: np.ndarray, face_box: Tuple[int, int, int, int]) -> Dict[str, float]:
        """Extract face deformation metrics."""
        features = {}
        
        if len(landmarks) < 468:
            return self._default_deformation_features()
        
        x, y, w, h = face_box
        face_area = w * h
        
        lip_points = landmarks[61:291]
        nose_points = landmarks[1:11]
        left_eye_points = landmarks[33:133]
        right_eye_points = landmarks[362:462]
        
        def calculate_irregularity(points):
            if len(points) < 3:
                return 0.0
            distances = []
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    distances.append(euclidean(points[i], points[j]))
            return float(np.std(distances)) if distances else 0.0
        
        features['lip_irregularity'] = calculate_irregularity(lip_points)
        features['nose_irregularity'] = calculate_irregularity(nose_points)
        features['left_eye_irregularity'] = calculate_irregularity(left_eye_points)
        features['right_eye_irregularity'] = calculate_irregularity(right_eye_points)
        
        nose_to_chin = euclidean(landmarks[4], landmarks[152])
        nose_to_forehead = euclidean(landmarks[4], landmarks[10])
        if nose_to_forehead > 0:
            features['face_proportion_ratio'] = float(nose_to_chin / nose_to_forehead)
        else:
            features['face_proportion_ratio'] = 1.0
        
        return features

    def _default_deformation_features(self) -> Dict[str, float]:
        return {
            'lip_irregularity': 0.0,
            'nose_irregularity': 0.0,
            'left_eye_irregularity': 0.0,
            'right_eye_irregularity': 0.0,
            'face_proportion_ratio': 1.0,
        }

    def extract_all_features(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Extract all face features from video frames."""
        logger.info(f"Extracting face features from {len(frames)} frames...")
        
        all_features = []
        frame_count = 0
        successful_frames = 0
        
        for frame in frames:
            if frame is None or len(frame.shape) != 3:
                continue
            
            frame_count += 1
            
            result = self.detect_face(frame)
            if result is None:
                continue
            
            successful_frames += 1
            features = {}
            
            eye_feats = self.extract_eye_features(result.landmarks)
            features.update(eye_feats)
            
            sym_feats = self.extract_symmetry_features(result.landmarks)
            features.update(sym_feats)
            
            skin_feats = self.extract_skin_texture_features(frame, result.face_box)
            features.update(skin_feats)
            
            self.track_blink(result.landmarks)
            
            landmark_feats = self.analyze_landmark_consistency(result.landmarks)
            features.update(landmark_feats)
            
            deform_feats = self.extract_face_deformation_features(result.landmarks, result.face_box)
            features.update(deform_feats)
            
            all_features.append(features)
        
        blink_feats = self.detect_blinks()
        
        summary = self._aggregate_features(all_features)
        summary.update(blink_feats)
        
        summary['face_detection_rate'] = successful_frames / max(frame_count, 1)
        summary['frames_analyzed'] = successful_frames
        
        logger.info(f"Face analysis complete: {successful_frames}/{frame_count} frames")
        
        return summary

    def _aggregate_features(self, features_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate features across frames."""
        if not features_list:
            return self._default_all_features()
        
        aggregated = {}
        feature_keys = features_list[0].keys()
        
        for key in feature_keys:
            values = [f[key] for f in features_list if key in f]
            if values:
                aggregated[f'{key}_mean'] = float(np.mean(values))
                aggregated[f'{key}_std'] = float(np.std(values))
                aggregated[f'{key}_min'] = float(np.min(values))
                aggregated[f'{key}_max'] = float(np.max(values))
        
        return aggregated

    def _default_all_features(self) -> Dict[str, float]:
        """Return default features when no face detected."""
        defaults = {}
        default_pairs = [
            ('eye_aspect_ratio_mean', 0.25),
            ('eye_aspect_ratio_std', 0.05),
            ('eye_distance_ratio', 0.3),
            ('face_symmetry_score', 1.0),
            ('face_symmetry_std', 0.0),
            ('skin_texture_variance', 500.0),
            ('skin_gradient_mean', 10.0),
            ('skin_gradient_std', 5.0),
            ('skin_edge_density', 0.1),
            ('landmark_temporal_variance', 0.0),
            ('landmark_temporal_variance_max', 0.0),
            ('avg_landmark_movement', 0.0),
            ('landmark_movement_std', 0.0),
            ('lip_irregularity', 0.0),
            ('nose_irregularity', 0.0),
            ('left_eye_irregularity', 0.0),
            ('right_eye_irregularity', 0.0),
            ('face_proportion_ratio', 1.0),
        ]
        
        for key, val in default_pairs:
            defaults[key] = val
            defaults[f'{key}_mean'] = val
            defaults[f'{key}_std'] = 0.0
            defaults[f'{key}_min'] = val
            defaults[f'{key}_max'] = val
        
        defaults['face_detection_rate'] = 0.0
        defaults['frames_analyzed'] = 0
        
        return defaults

    def reset(self):
        """Reset analyzer state."""
        self.blink_history = []
        self.landmark_history = []
