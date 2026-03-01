#!/usr/bin/env python3
# check_setup.py
"""
Script kiểm tra môi trường và dependencies
"""

import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Kiểm tra Python version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python version phải >= 3.8")
        print(f"   Hiện tại: Python {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_ffmpeg():
    """Kiểm tra FFmpeg"""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # Parse version
            version_line = result.stdout.split('\n')[0]
            print(f"✓ FFmpeg: {version_line}")
            return True
        else:
            print("❌ FFmpeg không hoạt động đúng")
            return False
    except FileNotFoundError:
        print("❌ FFmpeg không được tìm thấy")
        print("   Cài đặt:")
        print("   - Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("   - macOS: brew install ffmpeg")
        print("   - Windows: https://ffmpeg.org/download.html")
        return False
    except Exception as e:
        print(f"❌ Lỗi khi kiểm tra FFmpeg: {e}")
        return False


def check_package(package_name, import_name=None):
    """Kiểm tra Python package"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {package_name}: {version}")
        return True
    except ImportError:
        print(f"❌ {package_name} chưa được cài đặt")
        return False


def check_directories():
    """Kiểm tra và tạo thư mục cần thiết"""
    dirs = ['models', 'cache', 'logs', 'output', 'data']
    
    print("\nKiểm tra thư mục:")
    for dir_name in dirs:
        path = Path(dir_name)
        if path.exists():
            print(f"✓ {dir_name}/ đã tồn tại")
        else:
            path.mkdir(parents=True, exist_ok=True)
            print(f"✓ {dir_name}/ đã được tạo")
    
    return True


def check_config():
    """Kiểm tra file config"""
    config_file = Path('config.yaml')
    if config_file.exists():
        print(f"✓ config.yaml tồn tại")
        return True
    else:
        print(f"❌ config.yaml không tồn tại")
        print(f"   Tạo file config.yaml từ template")
        return False


def check_src_modules():
    """Kiểm tra các module trong src/"""
    modules = [
        'preprocessing',
        'forensic',
        'reality_engine',
        'stress_lab',
        'features',
        'classifier',
        'fusion',
        'utils'
    ]
    
    print("\nKiểm tra src modules:")
    all_ok = True
    
    for module_name in modules:
        module_file = Path(f'src/{module_name}.py')
        if module_file.exists():
            print(f"✓ src/{module_name}.py")
        else:
            print(f"❌ src/{module_name}.py không tồn tại")
            all_ok = False
    
    return all_ok


def main():
    print("=" * 80)
    print("HYBRID++ REALITY STRESS VIDEO AI DETECTOR")
    print("Environment Setup Check")
    print("=" * 80)
    
    checks = []
    
    # Check Python
    print("\n1. Kiểm tra Python:")
    checks.append(check_python_version())
    
    # Check FFmpeg
    print("\n2. Kiểm tra FFmpeg:")
    checks.append(check_ffmpeg())
    
    # Check dependencies
    print("\n3. Kiểm tra Python packages:")
    packages = [
        ('opencv-python', 'cv2'),
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('scikit-learn', 'sklearn'),
        ('lightgbm', 'lightgbm'),
        ('torch', 'torch'),
        ('pandas', 'pandas'),
        ('PyYAML', 'yaml'),
        ('matplotlib', 'matplotlib'),
        ('tqdm', 'tqdm'),
        ('joblib', 'joblib')
    ]
    
    for pkg_name, import_name in packages:
        checks.append(check_package(pkg_name, import_name))
    
    # Check directories
    checks.append(check_directories())
    
    # Check config
    print("\n4. Kiểm tra config:")
    checks.append(check_config())
    
    # Check src modules
    checks.append(check_src_modules())
    
    # Summary
    print("\n" + "=" * 80)
    if all(checks):
        print("✅ TẤT CẢ KIỂM TRA ĐỀU PASS!")
        print("\nBạn đã sẵn sàng để:")
        print("  1. Training: python train.py --data_dir data/")
        print("  2. Inference: python inference.py --video test.mp4")
    else:
        print("⚠️  MỘT SỐ KIỂM TRA THẤT BẠI")
        print("\nVui lòng:")
        print("  1. Cài đặt các packages thiếu: pip install -r requirements.txt")
        print("  2. Cài đặt FFmpeg nếu cần")
        print("  3. Chạy lại: python check_setup.py")
    print("=" * 80)


if __name__ == '__main__':
    main()