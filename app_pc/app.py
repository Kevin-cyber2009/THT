#!/usr/bin/env python3
# app.py
"""
Hybrid++ Video AI Detector - GUI Application
Giao diện kiểm tra video Real hay AI-generated
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import json
import threading

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QLabel, QTextEdit, QFileDialog,
    QProgressBar, QTabWidget, QTableWidget, QTableWidgetItem,
    QMessageBox, QGroupBox, QRadioButton, QComboBox, QFrame
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QFont, QIcon, QColor

# Import các module xử lý
from src.features import FeatureExtractor
from src.classifier import VideoClassifier
from src.fusion import ScoreFusion
from src.utils import load_config, setup_logging


class AnalysisThread(QThread):
    """Thread xử lý analysis để không block GUI"""
    
    finished = Signal(dict)  # Signal khi hoàn thành
    progress = Signal(str)   # Signal cập nhật progress
    error = Signal(str)      # Signal khi có lỗi
    
    def __init__(self, video_path, model_path, config_path):
        super().__init__()
        self.video_path = video_path
        self.model_path = model_path
        self.config_path = config_path
    
    def run(self):
        """Chạy analysis"""
        try:
            self.progress.emit("Đang khởi tạo...")
            
            # Load config
            config = load_config(self.config_path)
            
            # Load model
            self.progress.emit("Đang load model...")
            classifier = VideoClassifier(config)
            classifier.load(self.model_path)
            
            # Initialize extractors
            self.progress.emit("Đang khởi tạo feature extractor...")
            feature_extractor = FeatureExtractor(config)
            fusion_engine = ScoreFusion(config)
            
            # Extract features
            self.progress.emit("Đang trích xuất features từ video...")
            features_dict, metadata = feature_extractor.extract_from_video(self.video_path)
            
            # Get feature names
            feature_names = classifier.feature_names
            if feature_names is None:
                feature_names = feature_extractor.get_feature_names()
            
            # Convert to vector
            self.progress.emit("Đang xử lý features...")
            feature_vector = feature_extractor.features_to_vector(
                features_dict,
                feature_names
            )
            
            # Predict
            self.progress.emit("Đang phân tích...")
            X = feature_vector.reshape(1, -1)
            pred, prob = classifier.predict(X)
            
            # Calculate component scores
            self.progress.emit("Đang tính toán scores...")
            artifact_score = fusion_engine.compute_artifact_score(features_dict)
            reality_score = fusion_engine.compute_reality_score(features_dict)
            
            # Fusion
            fusion_result = fusion_engine.fuse_scores(
                artifact_score,
                reality_score,
                0.5  # Default stress score
            )
            
            # Generate explanations
            explanations = fusion_engine.generate_explanation(
                features_dict,
                fusion_result
            )
            
            # Prepare result
            result = {
                'video_path': self.video_path,
                'prediction': 'FAKE' if pred[0] == 1 else 'REAL',
                'probability_fake': float(prob[0]),
                'probability_real': float(1 - prob[0]),
                'artifact_score': float(artifact_score),
                'reality_score': float(reality_score),
                'fusion_result': fusion_result,
                'explanations': explanations,
                'metadata': metadata,
                'timestamp': datetime.now().isoformat()
            }
            
            self.progress.emit("Hoàn tất!")
            self.finished.emit(result)
            
        except Exception as e:
            self.error.emit(str(e))


class DownloadThread(QThread):
    """Thread download video từ URL"""
    
    finished = Signal(str)  # Signal trả về path của video đã download
    progress = Signal(str)
    error = Signal(str)
    
    def __init__(self, url, output_dir):
        super().__init__()
        self.url = url
        self.output_dir = output_dir
    
    def run(self):
        """Download video"""
        try:
            import subprocess
            import tempfile
            
            self.progress.emit("Đang download video...")
            
            # Create temp directory
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Download với yt-dlp
            output_template = os.path.join(self.output_dir, "downloaded_%(id)s.%(ext)s")
            
            cmd = [
                'yt-dlp',
                '-f', 'best[height<=1080]',
                '--max-filesize', '100M',
                '-o', output_template,
                self.url
            ]
            
            self.progress.emit("Đang xử lý...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode != 0:
                raise Exception(f"Download failed: {result.stderr}")
            
            # Tìm file đã download
            downloaded_files = list(Path(self.output_dir).glob("downloaded_*"))
            if not downloaded_files:
                raise Exception("Không tìm thấy file đã download")
            
            video_path = str(downloaded_files[-1])
            self.progress.emit("Download hoàn tất!")
            self.finished.emit(video_path)
            
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Hybrid++ Video AI Detector")
        self.setMinimumSize(1000, 700)
        
        # Paths
        self.model_path = "models/hybrid_detector.pkl"
        self.config_path = "config.yaml"
        self.history_file = "output/history.json"
        self.temp_dir = "temp"
        
        # Load history
        self.history = self.load_history()
        
        # Setup UI
        self.setup_ui()
        
        # Check model exists
        self.check_model()
    
    def setup_ui(self):
        """Setup giao diện"""
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Title
        title = QLabel("🎬 HYBRID++ VIDEO AI DETECTOR")
        title.setFont(QFont("Arial", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #2196F3; padding: 10px;")
        layout.addWidget(title)
        
        # Tabs
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # Tab 1: Analysis
        analysis_tab = self.create_analysis_tab()
        tabs.addTab(analysis_tab, "🔍 Phân tích")
        
        # Tab 2: History
        history_tab = self.create_history_tab()
        tabs.addTab(history_tab, "📊 Lịch sử")
        
        # Tab 3: Settings
        settings_tab = self.create_settings_tab()
        tabs.addTab(settings_tab, "⚙️ Cài đặt")
        
        # Status bar
        self.statusBar().showMessage("Sẵn sàng")
    
    def create_analysis_tab(self):
        """Tạo tab phân tích"""
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Input group
        input_group = QGroupBox("📥 Input Video")
        input_layout = QVBoxLayout()
        
        # Radio buttons
        radio_layout = QHBoxLayout()
        self.radio_file = QRadioButton("File trên máy")
        self.radio_url = QRadioButton("Link (YouTube, Facebook, ...)")
        self.radio_file.setChecked(True)
        self.radio_file.toggled.connect(self.on_input_type_changed)
        radio_layout.addWidget(self.radio_file)
        radio_layout.addWidget(self.radio_url)
        radio_layout.addStretch()
        input_layout.addLayout(radio_layout)
        
        # Input field
        input_field_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Chọn file video hoặc nhập URL...")
        self.btn_browse = QPushButton("📂 Chọn file")
        self.btn_browse.clicked.connect(self.browse_file)
        input_field_layout.addWidget(self.input_field)
        input_field_layout.addWidget(self.btn_browse)
        input_layout.addLayout(input_field_layout)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Analyze button
        self.btn_analyze = QPushButton("🔍 PHÂN TÍCH VIDEO")
        self.btn_analyze.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 12px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
        """)
        self.btn_analyze.clicked.connect(self.start_analysis)
        layout.addWidget(self.btn_analyze)
        
        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel()
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.hide()
        layout.addWidget(self.progress_label)
        
        # Result group
        result_group = QGroupBox("📊 Kết quả")
        result_layout = QVBoxLayout()
        
        # Result display
        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        self.result_display.setMinimumHeight(300)
        self.result_display.setStyleSheet("""
            QTextEdit {
                font-size: 14px;
                background-color: #F5F5F5;
                border: 1px solid #BDBDBD;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        result_layout.addWidget(self.result_display)
        
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)
        
        layout.addStretch()
        
        return widget
    
    def create_history_tab(self):
        """Tạo tab lịch sử"""
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Toolbar
        toolbar = QHBoxLayout()
        btn_refresh = QPushButton("🔄 Làm mới")
        btn_refresh.clicked.connect(self.refresh_history)
        btn_clear = QPushButton("🗑️ Xóa lịch sử")
        btn_clear.clicked.connect(self.clear_history)
        toolbar.addWidget(btn_refresh)
        toolbar.addWidget(btn_clear)
        toolbar.addStretch()
        layout.addLayout(toolbar)
        
        # History table
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels([
            "Thời gian", "Video", "Kết quả", "Confidence", "Chi tiết"
        ])
        self.history_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.history_table)
        
        # Load history
        self.refresh_history()
        
        return widget
    
    def create_settings_tab(self):
        """Tạo tab cài đặt"""
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Model settings
        model_group = QGroupBox("🤖 Model")
        model_layout = QVBoxLayout()
        
        model_path_layout = QHBoxLayout()
        model_path_layout.addWidget(QLabel("Model path:"))
        self.model_path_input = QLineEdit(self.model_path)
        btn_browse_model = QPushButton("Browse")
        btn_browse_model.clicked.connect(self.browse_model)
        model_path_layout.addWidget(self.model_path_input)
        model_path_layout.addWidget(btn_browse_model)
        model_layout.addLayout(model_path_layout)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Config settings
        config_group = QGroupBox("⚙️ Configuration")
        config_layout = QVBoxLayout()
        
        config_path_layout = QHBoxLayout()
        config_path_layout.addWidget(QLabel("Config path:"))
        self.config_path_input = QLineEdit(self.config_path)
        btn_browse_config = QPushButton("Browse")
        btn_browse_config.clicked.connect(self.browse_config)
        config_path_layout.addWidget(self.config_path_input)
        config_path_layout.addWidget(btn_browse_config)
        config_layout.addLayout(config_path_layout)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # Apply button
        btn_apply = QPushButton("✅ Áp dụng")
        btn_apply.clicked.connect(self.apply_settings)
        layout.addWidget(btn_apply)
        
        layout.addStretch()
        
        # About section
        about = QLabel("""
        <h3>Về Hybrid++ Detector</h3>
        <p>Version 1.0.0</p>
        <p>Hệ thống phát hiện video AI-generated sử dụng:</p>
        <ul>
            <li>Forensic Analysis (FFT, DCT, PRNU, Optical Flow)</li>
            <li>Reality Compliance Engine</li>
            <li>Adversarial Stress Testing</li>
        </ul>
        """)
        about.setWordWrap(True)
        layout.addWidget(about)
        
        return widget
    
    def on_input_type_changed(self):
        """Xử lý khi đổi input type"""
        if self.radio_file.isChecked():
            self.btn_browse.setEnabled(True)
            self.input_field.setPlaceholderText("Chọn file video hoặc nhập đường dẫn...")
        else:
            self.btn_browse.setEnabled(False)
            self.input_field.setPlaceholderText("Nhập URL (YouTube, Facebook, ...)...")
    
    def browse_file(self):
        """Chọn file video"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)"
        )
        if file_path:
            self.input_field.setText(file_path)
    
    def browse_model(self):
        """Chọn model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn model",
            "models",
            "Model Files (*.pkl);;All Files (*)"
        )
        if file_path:
            self.model_path_input.setText(file_path)
    
    def browse_config(self):
        """Chọn config file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn config",
            "",
            "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if file_path:
            self.config_path_input.setText(file_path)
    
    def apply_settings(self):
        """Áp dụng settings"""
        self.model_path = self.model_path_input.text()
        self.config_path = self.config_path_input.text()
        
        QMessageBox.information(self, "Thành công", "Đã lưu settings!")
        self.check_model()
    
    def check_model(self):
        """Kiểm tra model có tồn tại không"""
        if not os.path.exists(self.model_path):
            QMessageBox.warning(
                self,
                "Model không tồn tại",
                f"Không tìm thấy model tại: {self.model_path}\n\n"
                "Vui lòng train model trước hoặc chọn model khác trong Settings."
            )
            self.btn_analyze.setEnabled(False)
        else:
            self.btn_analyze.setEnabled(True)
    
    def start_analysis(self):
        """Bắt đầu phân tích"""
        
        input_text = self.input_field.text().strip()
        
        if not input_text:
            QMessageBox.warning(self, "Lỗi", "Vui lòng nhập file path hoặc URL!")
            return
        
        # Disable button
        self.btn_analyze.setEnabled(False)
        self.progress_bar.show()
        self.progress_label.show()
        
        if self.radio_url.isChecked():
            # Download video trước
            self.start_download(input_text)
        else:
            # Phân tích trực tiếp
            if not os.path.exists(input_text):
                QMessageBox.warning(self, "Lỗi", f"File không tồn tại: {input_text}")
                self.btn_analyze.setEnabled(True)
                self.progress_bar.hide()
                self.progress_label.hide()
                return
            
            self.analyze_video(input_text)
    
    def start_download(self, url):
        """Download video từ URL"""
        
        self.download_thread = DownloadThread(url, self.temp_dir)
        self.download_thread.progress.connect(self.update_progress)
        self.download_thread.finished.connect(self.on_download_finished)
        self.download_thread.error.connect(self.on_error)
        self.download_thread.start()
    
    def on_download_finished(self, video_path):
        """Xử lý khi download xong"""
        self.analyze_video(video_path)
    
    def analyze_video(self, video_path):
        """Phân tích video"""
        
        self.analysis_thread = AnalysisThread(
            video_path,
            self.model_path,
            self.config_path
        )
        self.analysis_thread.progress.connect(self.update_progress)
        self.analysis_thread.finished.connect(self.on_analysis_finished)
        self.analysis_thread.error.connect(self.on_error)
        self.analysis_thread.start()
    
    def update_progress(self, message):
        """Cập nhật progress"""
        self.progress_label.setText(message)
        self.statusBar().showMessage(message)
    
    def on_analysis_finished(self, result):
        """Xử lý khi phân tích xong"""
        
        # Hide progress
        self.progress_bar.hide()
        self.progress_label.hide()
        self.btn_analyze.setEnabled(True)
        
        # Display result
        self.display_result(result)
        
        # Save to history
        self.save_to_history(result)
        self.refresh_history()
        
        self.statusBar().showMessage("Phân tích hoàn tất!", 5000)
    
    def on_error(self, error_msg):
        """Xử lý lỗi"""
        
        self.progress_bar.hide()
        self.progress_label.hide()
        self.btn_analyze.setEnabled(True)
        
        QMessageBox.critical(self, "Lỗi", f"Có lỗi xảy ra:\n{error_msg}")
        self.statusBar().showMessage("Lỗi!", 5000)
    
    def display_result(self, result):
        """Hiển thị kết quả"""
        
        prediction = result['prediction']
        prob_fake = result['probability_fake']
        prob_real = result['probability_real']
        
        # Color based on prediction
        if prediction == 'FAKE':
            color = "#F44336"  # Red
            icon = "🔴"
            confidence = prob_fake
        else:
            color = "#4CAF50"  # Green
            icon = "🟢"
            confidence = prob_real
        
        # Build HTML
        html = f"""
        <div style="font-family: Arial;">
            <h2 style="color: {color}; text-align: center;">
                {icon} KẾT QUẢ: {prediction}
            </h2>
            
            <div style="background: #F5F5F5; padding: 15px; border-radius: 5px; margin: 10px 0;">
                <h3>📊 Confidence:</h3>
                <div style="background: white; padding: 10px; border-radius: 3px;">
                    <p><strong>Probability FAKE:</strong> {prob_fake:.1%}</p>
                    <p><strong>Probability REAL:</strong> {prob_real:.1%}</p>
                    <p><strong>Confidence Level:</strong> {result['fusion_result'].get('confidence', 'N/A')}</p>
                </div>
            </div>
            
            <div style="background: #F5F5F5; padding: 15px; border-radius: 5px; margin: 10px 0;">
                <h3>🔍 Component Scores:</h3>
                <div style="background: white; padding: 10px; border-radius: 3px;">
                    <p><strong>Artifact Score:</strong> {result['artifact_score']:.3f}</p>
                    <p><strong>Reality Score:</strong> {result['reality_score']:.3f}</p>
                </div>
            </div>
            
            <div style="background: #F5F5F5; padding: 15px; border-radius: 5px; margin: 10px 0;">
                <h3>💡 Giải thích:</h3>
                <ol>
        """
        
        for explanation in result['explanations']:
            html += f"<li>{explanation}</li>"
        
        html += """
                </ol>
            </div>
            
            <div style="background: #F5F5F5; padding: 15px; border-radius: 5px; margin: 10px 0;">
                <h3>📹 Video Info:</h3>
                <div style="background: white; padding: 10px; border-radius: 3px;">
        """
        
        metadata = result.get('metadata', {})
        html += f"<p><strong>Frames:</strong> {metadata.get('num_frames', 'N/A')}</p>"
        html += f"<p><strong>Resolution:</strong> {metadata.get('width', 'N/A')}x{metadata.get('height', 'N/A')}</p>"
        html += f"<p><strong>FPS:</strong> {metadata.get('fps', 'N/A'):.1f}</p>"
        html += f"<p><strong>Duration:</strong> {metadata.get('duration', 0):.1f}s</p>"
        
        html += """
                </div>
            </div>
        </div>
        """
        
        self.result_display.setHtml(html)
    
    def load_history(self):
        """Load lịch sử"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def save_to_history(self, result):
        """Lưu vào lịch sử"""
        # Simplify result for history
        history_entry = {
            'timestamp': result['timestamp'],
            'video_path': result['video_path'],
            'prediction': result['prediction'],
            'probability_fake': result['probability_fake'],
            'confidence': result['fusion_result'].get('confidence', 'N/A')
        }
        
        self.history.insert(0, history_entry)  # Newest first
        
        # Keep only last 100
        self.history = self.history[:100]
        
        # Save
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
    
    def refresh_history(self):
        """Làm mới bảng lịch sử"""
        self.history = self.load_history()
        
        self.history_table.setRowCount(len(self.history))
        
        for i, entry in enumerate(self.history):
            # Timestamp
            ts = datetime.fromisoformat(entry['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            self.history_table.setItem(i, 0, QTableWidgetItem(ts))
            
            # Video
            video_name = os.path.basename(entry['video_path'])
            self.history_table.setItem(i, 1, QTableWidgetItem(video_name))
            
            # Prediction
            pred_item = QTableWidgetItem(entry['prediction'])
            if entry['prediction'] == 'FAKE':
                pred_item.setForeground(QColor('#F44336'))
            else:
                pred_item.setForeground(QColor('#4CAF50'))
            self.history_table.setItem(i, 2, pred_item)
            
            # Confidence
            conf = f"{entry['probability_fake']:.1%}" if entry['prediction'] == 'FAKE' else f"{1-entry['probability_fake']:.1%}"
            self.history_table.setItem(i, 3, QTableWidgetItem(conf))
            
            # Detail
            detail = f"{entry['confidence']}"
            self.history_table.setItem(i, 4, QTableWidgetItem(detail))
    
    def clear_history(self):
        """Xóa lịch sử"""
        reply = QMessageBox.question(
            self,
            "Xác nhận",
            "Bạn có chắc muốn xóa toàn bộ lịch sử?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.history = []
            if os.path.exists(self.history_file):
                os.remove(self.history_file)
            self.refresh_history()
            QMessageBox.information(self, "Thành công", "Đã xóa lịch sử!")


def main():
    """Main function"""
    app = QApplication(sys.argv)
    
    # Set style
    app.setStyle('Fusion')
    
    # Create window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()