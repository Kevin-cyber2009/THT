# src/utils.py
"""
Module utilities chung cho toàn hệ thống
"""

import os
import logging
import yaml
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np


def setup_logging(config: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    Thiết lập logging cho hệ thống
    
    Args:
        config: Dictionary cấu hình logging
        
    Returns:
        Logger instance đã được config
    """
    if config is None:
        config = load_config()
    
    log_config = config.get('logging', {})
    log_dir = Path(log_config.get('log_dir', 'logs'))
    log_dir.mkdir(exist_ok=True)
    
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_file = log_dir / log_config.get('log_file', 'detector.log')
    
    # Tạo logger
    logger = logging.getLogger('hybrid_detector')
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(log_level)
    
    # Console handler
    if log_config.get('console_output', True):
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        logger.addHandler(ch)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    
    logger.addHandler(fh)
    
    return logger


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load cấu hình từ file YAML
    
    Args:
        config_path: Đường dẫn đến file config
        
    Returns:
        Dictionary chứa cấu hình
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file không tồn tại: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def check_ffmpeg() -> bool:
    """
    Kiểm tra xem FFmpeg đã được cài đặt chưa
    
    Returns:
        True nếu FFmpeg khả dụng, False nếu không
        
    Raises:
        RuntimeError: Nếu FFmpeg không tìm thấy
    """
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        raise RuntimeError(
            "FFmpeg không được tìm thấy. Vui lòng cài đặt FFmpeg:\n"
            "Ubuntu/Debian: sudo apt-get install ffmpeg\n"
            "Windows: Tải từ https://ffmpeg.org/download.html\n"
            "macOS: brew install ffmpeg"
        )


def ensure_dir(path: str) -> Path:
    """
    Tạo thư mục nếu chưa tồn tại
    
    Args:
        path: Đường dẫn thư mục
        
    Returns:
        Path object của thư mục
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_video_info(video_path: str) -> Dict[str, Any]:
    """
    Lấy thông tin metadata của video bằng FFprobe
    
    Args:
        video_path: Đường dẫn video
        
    Returns:
        Dictionary chứa thông tin video (fps, duration, resolution, etc.)
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            return {}
        
        import json
        data = json.loads(result.stdout)
        
        # Tìm video stream
        video_stream = None
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_stream = stream
                break
        
        if not video_stream:
            return {}
        
        info = {
            'width': int(video_stream.get('width', 0)),
            'height': int(video_stream.get('height', 0)),
            'fps': eval(video_stream.get('r_frame_rate', '0/1')),
            'duration': float(data.get('format', {}).get('duration', 0)),
            'codec': video_stream.get('codec_name', 'unknown'),
        }
        
        return info
        
    except Exception as e:
        logging.getLogger('hybrid_detector').warning(
            f"Không thể lấy thông tin video: {e}"
        )
        return {}


def normalize_array(arr: np.ndarray, method: str = 'standard') -> np.ndarray:
    """
    Chuẩn hóa numpy array
    
    Args:
        arr: Array cần chuẩn hóa
        method: Phương pháp ('standard', 'minmax', 'none')
        
    Returns:
        Array đã được chuẩn hóa
    """
    if method == 'none':
        return arr
    
    if method == 'standard':
        mean = np.mean(arr)
        std = np.std(arr)
        if std == 0:
            return arr - mean
        return (arr - mean) / std
    
    elif method == 'minmax':
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val == min_val:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)
    
    else:
        raise ValueError(f"Phương pháp chuẩn hóa không hợp lệ: {method}")


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """
    Chia an toàn, tránh division by zero
    
    Args:
        a: Số bị chia
        b: Số chia
        default: Giá trị mặc định nếu b = 0
        
    Returns:
        Kết quả phép chia hoặc default
    """
    if b == 0 or np.isnan(b) or np.isinf(b):
        return default
    return a / b


def clip_value(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Giới hạn giá trị trong khoảng [min_val, max_val]
    
    Args:
        value: Giá trị cần clip
        min_val: Giá trị min
        max_val: Giá trị max
        
    Returns:
        Giá trị đã được clip
    """
    return max(min_val, min(max_val, value))