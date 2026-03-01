# README.md

# Hybrid++ Reality Stress Video AI Detector

Hệ thống phát hiện video AI-generated dựa trên ba trụ cột:
1. **Forensic Analysis**: FFT/DCT spectrum, PRNU residual, Optical Flow
2. **Reality Compliance**: Multi-scale entropy, Fractal dimension, Causal motion, Information conservation
3. **Adversarial Stress Testing**: Physics perturbations, Compression cascade, Temporal shuffle

## Tính năng chính

- ✅ Phát hiện deepfake video không cần watermark
- ✅ Giải thích kết quả bằng 3 bullets dễ hiểu
- ✅ Stress testing với các nhiễu loạn vật lý
- ✅ UI desktop đơn giản với PySide6
- ✅ Xuất báo cáo PDF với biểu đồ
- ✅ Train model trên dataset nhỏ (50-200 videos)

## Yêu cầu hệ thống

- **OS**: Linux, Windows, macOS
- **Python**: 3.10+
- **FFmpeg**: Cài đặt sẵn trong PATH
- **RAM**: Tối thiểu 4GB
- **CPU**: Dual-core trở lên (GPU optional)

## Cài đặt

### Bước 1: Clone repository
```bash
git clone https://github.com/yourusername/hybrid_reality_detector.git
cd hybrid_reality_detector
```

### Bước 2: Tạo môi trường ảo
```bash
python -m venv venv

# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### Bước 3: Cài đặt dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Bước 4: Kiểm tra FFmpeg
```bash
ffmpeg -version
```

Nếu chưa có FFmpeg:
- **Ubuntu/Debian**: `sudo apt-get install ffmpeg`
- **Windows**: Tải từ https://ffmpeg.org/download.html và thêm vào PATH
- **macOS**: `brew install ffmpeg`

### Bước 5: Tạo thư mục cần thiết
```bash
mkdir -p models logs output data/synthetic/videos
```

## Sử dụng nhanh

### 1. Tạo dữ liệu synthetic để test
```bash
python data/synthetic/generate_data.py
```

Tạo ~20 video mẫu trong `data/synthetic/videos/`:
- 10 video "real-like" (chuyển động smooth)
- 10 video "fake-like" (artifacts giả lập)

### 2. Train classifier
```bash
python train_classifier.py --data data/synthetic/videos --output models/detector.pkl
```

Output:
- `models/detector.pkl`: Model đã train
- `logs/training.log`: Training logs

### 3. Chạy demo phân tích video
```bash
python run_demo.py --video data/synthetic/videos/real_0.mp4
```

Output:
- `output/result_real_0.json`: Kết quả phân tích
- `output/report_real_0.pdf`: Báo cáo PDF
- Console: In ra scores và prediction

### 4. Chạy GUI
```bash
python -m app.main_ui
```

Giao diện cho phép:
- Upload video
- Xem real-time analysis
- Hiển thị Artifact/Reality/Stress scores
- Xem 3 explanation bullets

## Cấu trúc Output

### JSON Result (`output/result_*.json`)
```json
{
  "version": "1.0.0",
  "timestamp": "2024-02-10T15:30:45",
  "video_path": "sample.mp4",
  "prediction": "FAKE",
  "confidence": "HIGH",
  "final_probability": 0.87,
  "scores": {
    "artifact_score": 0.75,
    "reality_score": 0.32,
    "stress_score": 0.41
  },
  "features": {
    "fft_mean": 2.34,
    "entropy_slope": -0.15,
    ...
  },
  "explanations": [
    "Phát hiện artifacts: Phổ FFT có độ lệch chuẩn cao (0.087), PRNU autocorrelation thấp (0.234)",
    "Vi phạm reality compliance: Entropy slope bất thường (0.23), fractal dimension 1.12",
    "Không ổn định dưới perturbations (stability 0.41), features thay đổi mạnh khi bị stress test"
  ],
  "metadata": {
    "num_frames": 48,
    "fps": 6,
    "duration": 8.0
  }
}
```

## Chạy Tests
```bash
# Tất cả tests
pytest tests/ -v

# Test nhanh
pytest tests/ -q

# Test cụ thể
pytest tests/test_pipeline.py -v

# Coverage
pytest tests/ --cov=src --cov-report=html
```

## CLI Commands

### Extract features từ video
```bash
python -m src.features --video sample.mp4 --output features.json
```

### Train model
```bash
python train_classifier.py \
    --data data/synthetic/videos \
    --output models/detector.pkl \
    --cv-folds 5
```

### Predict video
```bash
python run_demo.py \
    --video sample.mp4 \
    --model models/detector.pkl \
    --output output/
```

### Run stress tests only
```bash
python -m src.stress_lab --video sample.mp4 --output stress_results.json
```

## Cấu hình

Chỉnh sửa `config.yaml` để thay đổi:
```yaml
preprocessing:
  fps: 6              # Frame sampling rate
  resize_width: 512
  max_frames: 100

forensic:
  fft_components: 10
  prnu_method: 'bilateral'

reality_engine:
  entropy_scales: 4
  fractal_box_sizes: [2, 4, 8, 16, 32]

stress_lab:
  light_jitter_strength: 0.1
  compression_levels: [23, 28, 35]

fusion:
  artifact_weight: 0.4
  reality_weight: 0.35
  stress_weight: 0.25
  threshold_fake: 0.5
```

## Docker (Optional)
```bash
# Build image
docker build -t hybrid-detector .

# Run container
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  hybrid-detector \
  python run_demo.py --video data/sample.mp4
```

## Troubleshooting

### Lỗi: "FFmpeg không được tìm thấy"

→ Cài đặt FFmpeg và thêm vào PATH

### Lỗi: "Video quá ngắn"

→ Video cần >= 10 frames. Hệ thống sẽ tự động padding nhưng kết quả có thể không chính xác.

### Warning: "Video không có motion"

→ Optical flow features sẽ thấp nhưng không ảnh hưởng nghiêm trọng đến kết quả.

### Tests chạy chậm

→ Giảm `max_frames` trong config xuống 30-50 cho testing nhanh hơn.

## Giới hạn

- **PRNU**: Chỉ là residual approximation, không phải full sensor fingerprint
- **Fractal**: Box-counting proxy, không phải multifractal spectrum
- **Causal**: Linear predictor đơn giản, TCN nặng hơn có thể tốt hơn
- **Dataset**: Synthetic data chỉ để demo, cần real deepfake dataset để production

## Roadmap

- [ ] Thêm hỗ trợ temporal transformer cho causal motion
- [ ] Tích hợp face-specific deepfake detectors
- [ ] Hỗ trợ batch processing nhiều videos
- [ ] API REST endpoint
- [ ] Web interface

## Tham khảo

- FFT/DCT forensics: Farid (2009)
- PRNU: Lukas et al. (2006)
- Fractal analysis: Pentland (1984)
- Adversarial robustness: Carlini & Wagner (2017)

## License

MIT License - Dự án nghiên cứu sinh viên

## Tác giả

Student Research Project - 2024

