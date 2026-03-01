# src/preprocessing.py
"""
Module preprocessing: Trích xuất và tiền xử lý frames từ video
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging

from .utils import load_config, get_video_info


logger = logging.getLogger('hybrid_detector.preprocessing')


class VideoPreprocessor:
    """
    Class xử lý video: trích frames, resize, normalization
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Khởi tạo VideoPreprocessor
        
        Args:
            config: Dictionary cấu hình preprocessing
        """
        if config is None:
            config = load_config()
        
        self.preproc_config = config.get('preprocessing', {})
        self.fps = self.preproc_config.get('fps', 6)
        self.resize_width = self.preproc_config.get('resize_width', 512)
        self.resize_height = self.preproc_config.get('resize_height', 288)
        self.max_frames = self.preproc_config.get('max_frames', 100)
        self.color_space = self.preproc_config.get('color_space', 'RGB')
        
        logger.info(f"VideoPreprocessor initialized: fps={self.fps}, "
                   f"resize=({self.resize_width}, {self.resize_height})")
    
    def extract_frames(self, video_path: str) -> np.ndarray:
        """
        Trích xuất frames từ video với sampling fps
        
        Args:
            video_path: Đường dẫn đến file video
            
        Returns:
            Array shape (N, H, W, C) chứa frames
            
        Raises:
            FileNotFoundError: Nếu video không tồn tại
            ValueError: Nếu không đọc được video
        """
        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"Video không tồn tại: {video_path}")
        
        logger.info(f"Extracting frames từ: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Không thể mở video: {video_path}")
        
        # Lấy thông tin video
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if original_fps == 0:
            logger.warning("FPS không xác định, sử dụng 30 fps")
            original_fps = 30
        
        # Tính frame interval để sampling
        frame_interval = max(1, int(original_fps / self.fps))
        
        frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Chỉ lấy frames theo interval
            if frame_idx % frame_interval == 0:
                # Resize frame
                resized = cv2.resize(
                    frame,
                    (self.resize_width, self.resize_height),
                    interpolation=cv2.INTER_LINEAR
                )
                
                # Convert color space
                if self.color_space == 'RGB':
                    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                elif self.color_space == 'GRAY':
                    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                    resized = np.expand_dims(resized, axis=-1)
                
                frames.append(resized)
                
                # Giới hạn số frames
                if len(frames) >= self.max_frames:
                    logger.info(f"Đạt giới hạn {self.max_frames} frames")
                    break
            
            frame_idx += 1
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError("Không trích xuất được frame nào từ video")
        
        frames_array = np.array(frames, dtype=np.uint8)
        logger.info(f"Extracted {len(frames)} frames, shape: {frames_array.shape}")
        
        return frames_array
    
    def normalize_frames(self, frames: np.ndarray) -> np.ndarray:
        """
        Normalize frames về range [0, 1]
        
        Args:
            frames: Array frames shape (N, H, W, C)
            
        Returns:
            Normalized frames
        """
        return frames.astype(np.float32) / 255.0
    
    def preprocess(self, video_path: str) -> Tuple[np.ndarray, dict]:
        """
        Pipeline đầy đủ: extract + normalize
        
        Args:
            video_path: Đường dẫn video
            
        Returns:
            Tuple (normalized_frames, metadata)
        """
        frames = self.extract_frames(video_path)
        normalized = self.normalize_frames(frames)
        
        metadata = {
            'num_frames': len(frames),
            'shape': frames.shape,
            'sampling_fps': self.fps,
            'resize': (self.resize_width, self.resize_height),
        }
        
        # Thêm video info nếu có
        video_info = get_video_info(video_path)
        metadata.update(video_info)
        
        return normalized, metadata
    
    def handle_short_video(self, frames: np.ndarray, min_frames: int = 10) -> np.ndarray:
        """
        Xử lý video quá ngắn bằng cách repeat frames
        
        Args:
            frames: Array frames
            min_frames: Số frames tối thiểu cần thiết
            
        Returns:
            Frames đã được padding/repeat
        """
        num_frames = len(frames)
        
        if num_frames >= min_frames:
            return frames
        
        logger.warning(f"Video quá ngắn ({num_frames} frames), "
                      f"repeat để đạt {min_frames} frames")
        
        # Repeat frames để đạt min_frames
        repeat_times = (min_frames // num_frames) + 1
        repeated = np.tile(frames, (repeat_times, 1, 1, 1))
        
        return repeated[:min_frames]