#!/usr/bin/env python3
# final.py
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

# ============================================================
# FIX ĐƯỜNG DẪN - Hoạt động cả khi chạy .py lẫn .exe
# ============================================================
if getattr(sys, 'frozen', False):
    # Đang chạy là .exe (PyInstaller)
    BASE_PATH = os.path.dirname(sys.executable)
else:
    # Đang chạy là .py thường
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))

os.chdir(BASE_PATH)  # Set working directory về đúng chỗ
# ============================================================

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
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")


class DownloadThread(QThread):
    """Thread download video từ URL"""
    
    finished = Signal(str)
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
            
            self.progress.emit("Đang download video...")
            
            os.makedirs(self.output_dir, exist_ok=True)
            
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
        
        # ===== ĐƯỜNG DẪN - Tự động tìm đúng vị trí dù chạy .py hay .exe =====
        self.model_path = os.path.join(BASE_PATH, "models", "delta.pkl")
        self.config_path = os.path.join(BASE_PATH, "config.yaml")
        self.history_file = os.path.join(BASE_PATH, "output", "history.json")
        self.temp_dir = os.path.join(BASE_PATH, "temp")
        # ======================================================================
        
        # Set dark theme
        self.setup_theme()
        
        # Load history
        self.history = self.load_history()
        
        # Setup UI
        self.setup_ui()
        
        # Check model exists
        self.check_model()
    
    def setup_theme(self):
        """Setup dark theme"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1E1E1E;
            }
            QWidget {
                background-color: #1E1E1E;
                color: #E0E0E0;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QLabel {
                color: #E0E0E0;
            }
            QLineEdit {
                background-color: #2D2D2D;
                color: #FFFFFF;
                border: 1px solid #404040;
                border-radius: 4px;
                padding: 8px;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 1px solid #2196F3;
            }
            QPushButton {
                background-color: #2D2D2D;
                color: #FFFFFF;
                border: 1px solid #404040;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #3D3D3D;
                border: 1px solid #2196F3;
            }
            QPushButton:pressed {
                background-color: #1D1D1D;
            }
            QTextEdit {
                background-color: #2D2D2D;
                color: #FFFFFF;
                border: 1px solid #404040;
                border-radius: 4px;
                padding: 10px;
                font-size: 14px;
            }
            QGroupBox {
                border: 1px solid #404040;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 10px;
                color: #E0E0E0;
                font-weight: bold;
            }
            QGroupBox::title {
                color: #2196F3;
            }
            QRadioButton {
                color: #E0E0E0;
                spacing: 8px;
            }
            QRadioButton::indicator {
                width: 18px;
                height: 18px;
            }
            QRadioButton::indicator:unchecked {
                border: 2px solid #606060;
                border-radius: 9px;
                background-color: #2D2D2D;
            }
            QRadioButton::indicator:checked {
                border: 2px solid #2196F3;
                border-radius: 9px;
                background-color: #2196F3;
            }
            QTableWidget {
                background-color: #2D2D2D;
                color: #FFFFFF;
                gridline-color: #404040;
                border: 1px solid #404040;
                border-radius: 4px;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QTableWidget::item:selected {
                background-color: #2196F3;
            }
            QHeaderView::section {
                background-color: #3D3D3D;
                color: #FFFFFF;
                padding: 8px;
                border: none;
                border-right: 1px solid #404040;
                border-bottom: 1px solid #404040;
                font-weight: bold;
            }
            QProgressBar {
                background-color: #2D2D2D;
                border: 1px solid #404040;
                border-radius: 4px;
                text-align: center;
                color: #FFFFFF;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
                border-radius: 3px;
            }
            QTabWidget::pane {
                border: 1px solid #404040;
                background-color: #1E1E1E;
            }
            QTabBar::tab {
                background-color: #2D2D2D;
                color: #E0E0E0;
                padding: 10px 20px;
                border: 1px solid #404040;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #1E1E1E;
                color: #2196F3;
                border-bottom: 2px solid #2196F3;
            }
            QTabBar::tab:hover {
                background-color: #3D3D3D;
            }
            QStatusBar {
                background-color: #2D2D2D;
                color: #E0E0E0;
                border-top: 1px solid #404040;
            }
        """)
    
    def setup_ui(self):
        """Setup giao diện"""
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        title = QLabel("🎬 HYBRID++ VIDEO AI DETECTOR")
        title.setFont(QFont("Arial", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #2196F3; padding: 10px;")
        layout.addWidget(title)
        
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        analysis_tab = self.create_analysis_tab()
        tabs.addTab(analysis_tab, "🔍 Phân tích")
        
        history_tab = self.create_history_tab()
        tabs.addTab(history_tab, "📊 Lịch sử")
        
        about_tab = self.create_about_tab()
        tabs.addTab(about_tab, "ℹ️ Thông tin")
        
        self.statusBar().showMessage("Sẵn sàng")
    
    def create_analysis_tab(self):
        """Tạo tab phân tích"""
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        input_group = QGroupBox("📥 Input Video")
        input_layout = QVBoxLayout()
        
        radio_layout = QHBoxLayout()
        self.radio_file = QRadioButton("File trên máy")
        self.radio_url = QRadioButton("Link (YouTube, Facebook, ...)")
        self.radio_file.setChecked(True)
        self.radio_file.toggled.connect(self.on_input_type_changed)
        radio_layout.addWidget(self.radio_file)
        radio_layout.addWidget(self.radio_url)
        radio_layout.addStretch()
        input_layout.addLayout(radio_layout)
        
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
            QPushButton:hover { background-color: #1976D2; }
            QPushButton:disabled { background-color: #BDBDBD; }
        """)
        self.btn_analyze.clicked.connect(self.start_analysis)
        layout.addWidget(self.btn_analyze)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel()
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.hide()
        layout.addWidget(self.progress_label)
        
        result_group = QGroupBox("📊 Kết quả")
        result_layout = QVBoxLayout()
        
        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        self.result_display.setMinimumHeight(300)
        result_layout.addWidget(self.result_display)
        
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)
        
        layout.addStretch()
        
        return widget
    
    def create_history_tab(self):
        """Tạo tab lịch sử"""
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        toolbar = QHBoxLayout()
        btn_refresh = QPushButton("🔄 Làm mới")
        btn_refresh.clicked.connect(self.refresh_history)
        btn_clear = QPushButton("🗑️ Xóa lịch sử")
        btn_clear.clicked.connect(self.clear_history)
        toolbar.addWidget(btn_refresh)
        toolbar.addWidget(btn_clear)
        toolbar.addStretch()
        layout.addLayout(toolbar)
        
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels([
            "Thời gian", "Video", "Kết quả", "Confidence", "Chi tiết"
        ])
        self.history_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.history_table)
        
        self.refresh_history()
        
        return widget
    
    def create_about_tab(self):
        """Tạo tab thông tin"""
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        title = QLabel("🎬 HYBRID++ DETECTOR")
        title.setFont(QFont("Arial", 24, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #2196F3; padding: 20px;")
        layout.addWidget(title)
        
        version = QLabel("Version 1.0.0")
        version.setFont(QFont("Arial", 12))
        version.setAlignment(Qt.AlignCenter)
        version.setStyleSheet("color: #B0B0B0;")
        layout.addWidget(version)
        
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("background-color: #404040;")
        layout.addWidget(line)
        
        model_group = QGroupBox("📦 Cấu hình hiện tại")
        model_layout = QVBoxLayout()
        model_info = QLabel(f"""
        <div style='color: #E0E0E0; line-height: 1.8;'>
        <p><b>Model:</b> <span style='color: #2196F3;'>{self.model_path}</span></p>
        <p><b>Config:</b> <span style='color: #2196F3;'>{self.config_path}</span></p>
        <p><b>Base Path:</b> <span style='color: #2196F3;'>{BASE_PATH}</span></p>
        </div>
        """)
        model_info.setWordWrap(True)
        model_layout.addWidget(model_info)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        layout.addStretch()
        
        return widget
    
    def on_input_type_changed(self):
        if self.radio_file.isChecked():
            self.btn_browse.setEnabled(True)
            self.input_field.setPlaceholderText("Chọn file video hoặc nhập đường dẫn...")
        else:
            self.btn_browse.setEnabled(False)
            self.input_field.setPlaceholderText("Nhập URL (YouTube, Facebook, ...)...")
    
    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)"
        )
        if file_path:
            self.input_field.setText(file_path)
    
    def check_model(self):
        """Kiểm tra model có tồn tại không"""
        if not os.path.exists(self.model_path):
            QMessageBox.warning(
                self,
                "Model không tồn tại",
                f"Không tìm thấy model tại:\n{self.model_path}\n\n"
                f"Vui lòng đảm bảo file models/delta.pkl tồn tại."
            )
            self.btn_analyze.setEnabled(False)
        else:
            self.btn_analyze.setEnabled(True)
            self.statusBar().showMessage(f"✓ Model loaded: {self.model_path}")
    
    def start_analysis(self):
        input_text = self.input_field.text().strip()
        
        if not input_text:
            QMessageBox.warning(self, "Lỗi", "Vui lòng nhập file path hoặc URL!")
            return
        
        self.btn_analyze.setEnabled(False)
        self.progress_bar.show()
        self.progress_label.show()
        self.result_display.clear()
        
        if self.radio_url.isChecked():
            self.start_download(input_text)
        else:
            if not os.path.exists(input_text):
                QMessageBox.warning(self, "Lỗi", f"File không tồn tại:\n{input_text}")
                self.btn_analyze.setEnabled(True)
                self.progress_bar.hide()
                self.progress_label.hide()
                return
            self.analyze_video(input_text)
    
    def start_download(self, url):
        self.download_thread = DownloadThread(url, self.temp_dir)
        self.download_thread.progress.connect(self.update_progress)
        self.download_thread.finished.connect(self.on_download_finished)
        self.download_thread.error.connect(self.on_error)
        self.download_thread.start()
    
    def on_download_finished(self, video_path):
        self.analyze_video(video_path)
    
    def analyze_video(self, video_path):
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
        self.progress_label.setText(message)
        self.statusBar().showMessage(message)
    
    def on_analysis_finished(self, result):
        self.progress_bar.hide()
        self.progress_label.hide()
        self.btn_analyze.setEnabled(True)
        self.display_result(result)
        self.save_to_history(result)
        self.refresh_history()
        self.statusBar().showMessage("✓ Phân tích hoàn tất!", 5000)
    
    def on_error(self, error_msg):
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
        
        if prediction == 'FAKE':
            color = "#F44336"
            icon = "🔴"
            confidence = prob_fake
        else:
            color = "#4CAF50"
            icon = "🟢"
            confidence = prob_real
        
        html = f"""
        <div style="font-family: Arial; color: #212121;">
            <h2 style="color: {color}; text-align: center; font-size: 24px;">
                {icon} KẾT QUẢ: {prediction}
            </h2>
            
            <div style="background: #E3F2FD; padding: 15px; border-radius: 5px; margin: 10px 0;">
                <h3 style="color: #1565C0;">📊 Confidence:</h3>
                <p><strong>Probability FAKE:</strong> {prob_fake:.1%}</p>
                <p><strong>Probability REAL:</strong> {prob_real:.1%}</p>
                <p><strong>Confidence Level:</strong> {result['fusion_result'].get('confidence', 'N/A')}</p>
            </div>
            
            <div style="background: #F3E5F5; padding: 15px; border-radius: 5px; margin: 10px 0;">
                <h3 style="color: #6A1B9A;">🔍 Component Scores:</h3>
                <p><strong>Artifact Score:</strong> {result['artifact_score']:.3f}</p>
                <p><strong>Reality Score:</strong> {result['reality_score']:.3f}</p>
            </div>
            
            <div style="background: #E8F5E9; padding: 15px; border-radius: 5px; margin: 10px 0;">
                <h3 style="color: #2E7D32;">💡 Giải thích:</h3>
                <ol>
        """
        
        for explanation in result.get('explanations', []):
            html += f"<li style='margin: 5px 0;'>{explanation}</li>"
        
        html += """
                </ol>
            </div>
            
            <div style="background: #FFF3E0; padding: 15px; border-radius: 5px; margin: 10px 0;">
                <h3 style="color: #E65100;">📹 Video Info:</h3>
        """
        
        metadata = result.get('metadata', {})
        html += f"<p><strong>Frames:</strong> {metadata.get('num_frames', 'N/A')}</p>"
        html += f"<p><strong>Resolution:</strong> {metadata.get('width', 'N/A')}x{metadata.get('height', 'N/A')}</p>"
        
        fps = metadata.get('fps', 0)
        try:
            html += f"<p><strong>FPS:</strong> {float(fps):.1f}</p>"
        except:
            html += f"<p><strong>FPS:</strong> {fps}</p>"
        
        html += f"<p><strong>Duration:</strong> {metadata.get('duration', 0):.1f}s</p>"
        
        html += """
            </div>
        </div>
        """
        
        self.result_display.setHtml(html)
    
    def load_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def save_to_history(self, result):
        history_entry = {
            'timestamp': result['timestamp'],
            'video_path': result['video_path'],
            'prediction': result['prediction'],
            'probability_fake': result['probability_fake'],
            'confidence': result['fusion_result'].get('confidence', 'N/A')
        }
        
        self.history.insert(0, history_entry)
        self.history = self.history[:100]
        
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
    
    def refresh_history(self):
        self.history = self.load_history()
        self.history_table.setRowCount(len(self.history))
        
        for i, entry in enumerate(self.history):
            ts = datetime.fromisoformat(entry['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            self.history_table.setItem(i, 0, QTableWidgetItem(ts))
            
            video_name = os.path.basename(entry['video_path'])
            self.history_table.setItem(i, 1, QTableWidgetItem(video_name))
            
            pred_item = QTableWidgetItem(entry['prediction'])
            pred_item.setForeground(QColor('#F44336') if entry['prediction'] == 'FAKE' else QColor('#4CAF50'))
            self.history_table.setItem(i, 2, pred_item)
            
            prob = entry['probability_fake']
            conf = f"{prob:.1%}" if entry['prediction'] == 'FAKE' else f"{1-prob:.1%}"
            self.history_table.setItem(i, 3, QTableWidgetItem(conf))
            self.history_table.setItem(i, 4, QTableWidgetItem(entry['confidence']))
    
    def clear_history(self):
        reply = QMessageBox.question(
            self, "Xác nhận",
            "Bạn có chắc muốn xóa toàn bộ lịch sử?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.history = []
            if os.path.exists(self.history_file):
                os.remove(self.history_file)
            self.refresh_history()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()