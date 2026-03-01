# app/main_ui.py

"""
GUI Desktop Application với PySide6
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTextEdit, QProgressBar,
    QGroupBox, QGridLayout, QMessageBox, QFrame
)
from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QFont, QColor, QPalette

from src.utils import load_config, check_ffmpeg, setup_logging
from src.features import FeatureExtractor
from src.fusion import ScoreFusion
from src.preprocessing import VideoPreprocessor
from src.stress_lab import StressLab
from src.report import ReportGenerator


class AnalysisWorker(QThread):
    """
    Worker thread để chạy analysis không block UI
    """
    
    # Signals
    progress_update = Signal(int, str)  # (percentage, message)
    analysis_complete = Signal(dict)  # result dictionary
    analysis_error = Signal(str)  # error message
    
    def __init__(self, video_path: str, config: dict, run_stress: bool = True):
        super().__init__()
        self.video_path = video_path
        self.config = config
        self.run_stress = run_stress
    
    def run(self):
        """
        Chạy full pipeline analysis
        """
        try:
            # Initialize components
            self.progress_update.emit(5, "Khởi tạo components...")
            
            extractor = FeatureExtractor(self.config)
            fusion = ScoreFusion(self.config)
            
            # Extract features
            self.progress_update.emit(15, "Trích xuất frames từ video...")
            features, metadata = extractor.extract_from_video(self.video_path)
            
            self.progress_update.emit(40, f"Đã trích xuất {len(features)} features")
            
            # Run stress tests nếu được yêu cầu
            stress_results = None
            if self.run_stress:
                self.progress_update.emit(50, "Chạy stress tests (có thể mất vài phút)...")
                
                preprocessor = VideoPreprocessor(self.config)
                frames, _ = preprocessor.preprocess(self.video_path)
                
                stress_lab = StressLab(self.config)
                stress_results = stress_lab.run_stress_tests(frames, extractor)
                
                self.progress_update.emit(75, "Stress tests hoàn tất")
            else:
                stress_results = {'aggregate_stability_score': 0.5}
                self.progress_update.emit(60, "Bỏ qua stress tests")
            
            # Compute scores
            self.progress_update.emit(80, "Tính toán scores...")
            
            artifact_score = fusion.compute_artifact_score(features)
            reality_score = fusion.compute_reality_score(features)
            stress_score = fusion.compute_stress_score(stress_results)
            
            # Fuse scores
            self.progress_update.emit(90, "Kết hợp scores...")
            result = fusion.fuse_scores(artifact_score, reality_score, stress_score)
            
            # Generate explanations
            explanations = fusion.generate_explanation(features, result)
            
            # Prepare output
            self.progress_update.emit(95, "Hoàn tất phân tích")
            
            output_data = {
                'version': '1.0.0',
                'timestamp': datetime.now().isoformat(),
                'video_path': self.video_path,
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'final_probability': result['final_probability'],
                'scores': {
                    'artifact_score': artifact_score,
                    'reality_score': reality_score,
                    'stress_score': stress_score
                },
                'features': features,
                'explanations': explanations,
                'metadata': metadata
            }
            
            self.progress_update.emit(100, "Hoàn tất!")
            self.analysis_complete.emit(output_data)
            
        except Exception as e:
            self.analysis_error.emit(str(e))


class MainWindow(QMainWindow):
    """
    Main window của application
    """
    
    def __init__(self):
        super().__init__()
        
        self.video_path = None
        self.analysis_result = None
        self.config = None
        self.worker = None
        
        # Load config
        try:
            self.config = load_config()
        except Exception as e:
            QMessageBox.critical(
                self,
                "Lỗi Config",
                f"Không thể load config.yaml: {e}"
            )
            sys.exit(1)
        
        # Setup logging
        setup_logging(self.config)
        
        # Check FFmpeg
        try:
            check_ffmpeg()
        except RuntimeError as e:
            QMessageBox.critical(
                self,
                "Lỗi FFmpeg",
                str(e)
            )
            sys.exit(1)
        
        self.init_ui()
    
    def init_ui(self):
        """
        Khởi tạo UI components
        """
        self.setWindowTitle("Hybrid++ Reality Stress Video AI Detector")
        self.setGeometry(100, 100, 900, 700)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Title
        title_label = QLabel("Video AI Detection System")
        title_font = QFont("Arial", 20, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50; padding: 10px;")
        main_layout.addWidget(title_label)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator)
        
        # Video selection group
        video_group = QGroupBox("1. Chọn Video")
        video_layout = QHBoxLayout()
        
        self.video_label = QLabel("Chưa chọn video")
        self.video_label.setStyleSheet("padding: 5px; background-color: #ecf0f1; border-radius: 3px;")
        video_layout.addWidget(self.video_label)
        
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.setMinimumSize(QSize(120, 35))
        self.browse_btn.clicked.connect(self.browse_video)
        video_layout.addWidget(self.browse_btn)
        
        video_group.setLayout(video_layout)
        main_layout.addWidget(video_group)
        
        # Analysis options group
        options_group = QGroupBox("2. Tùy chọn")
        options_layout = QHBoxLayout()
        
        self.stress_checkbox = QPushButton("Bao gồm Stress Tests")
        self.stress_checkbox.setCheckable(True)
        self.stress_checkbox.setChecked(True)
        self.stress_checkbox.setMinimumSize(QSize(180, 35))
        self.stress_checkbox.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton:checked {
                background-color: #2ecc71;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        options_layout.addWidget(self.stress_checkbox)
        
        options_layout.addStretch()
        
        options_group.setLayout(options_layout)
        main_layout.addWidget(options_group)
        
        # Analysis button
        self.analyze_btn = QPushButton("▶ Bắt đầu Phân tích")
        self.analyze_btn.setMinimumSize(QSize(200, 50))
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self.start_analysis)
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        main_layout.addWidget(self.analyze_btn, alignment=Qt.AlignCenter)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimumHeight(25)
        main_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("")
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setStyleSheet("color: #7f8c8d; font-style: italic;")
        main_layout.addWidget(self.progress_label)
        
        # Results group
        results_group = QGroupBox("3. Kết quả")
        results_layout = QVBoxLayout()
        
        # Scores display
        scores_layout = QGridLayout()
        
        # Prediction label
        self.prediction_label = QLabel("Chưa có kết quả")
        self.prediction_label.setFont(QFont("Arial", 18, QFont.Bold))
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.prediction_label.setMinimumHeight(60)
        self.prediction_label.setStyleSheet("""
            background-color: #ecf0f1;
            border-radius: 8px;
            padding: 10px;
        """)
        scores_layout.addWidget(self.prediction_label, 0, 0, 1, 3)
        
        # Score labels
        self.artifact_label = self.create_score_label("Artifact Score", "--")
        self.reality_label = self.create_score_label("Reality Score", "--")
        self.stress_label = self.create_score_label("Stress Score", "--")
        
        scores_layout.addWidget(self.artifact_label, 1, 0)
        scores_layout.addWidget(self.reality_label, 1, 1)
        scores_layout.addWidget(self.stress_label, 1, 2)
        
        results_layout.addLayout(scores_layout)
        
        # Explanations
        exp_label = QLabel("Giải thích:")
        exp_label.setFont(QFont("Arial", 12, QFont.Bold))
        results_layout.addWidget(exp_label)
        
        self.explanation_text = QTextEdit()
        self.explanation_text.setReadOnly(True)
        self.explanation_text.setMinimumHeight(150)
        self.explanation_text.setStyleSheet("""
            background-color: #fdfefe;
            border: 1px solid #bdc3c7;
            border-radius: 5px;
            padding: 10px;
            font-size: 11px;
        """)
        results_layout.addWidget(self.explanation_text)
        
        results_group.setLayout(results_layout)
        main_layout.addWidget(results_group)
        
        # Action buttons
        action_layout = QHBoxLayout()
        
        self.save_json_btn = QPushButton("💾 Lưu JSON")
        self.save_json_btn.setMinimumSize(QSize(120, 35))
        self.save_json_btn.setEnabled(False)
        self.save_json_btn.clicked.connect(self.save_json)
        action_layout.addWidget(self.save_json_btn)
        
        self.save_pdf_btn = QPushButton("📄 Lưu PDF")
        self.save_pdf_btn.setMinimumSize(QSize(120, 35))
        self.save_pdf_btn.setEnabled(False)
        self.save_pdf_btn.clicked.connect(self.save_pdf)
        action_layout.addWidget(self.save_pdf_btn)
        
        action_layout.addStretch()
        
        main_layout.addLayout(action_layout)
        
        # Status bar
        self.statusBar().showMessage("Sẵn sàng")
    
    def create_score_label(self, title: str, value: str) -> QLabel:
        """
        Tạo label hiển thị score
        """
        label = QLabel(f"{title}\n{value}")
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumHeight(70)
        label.setStyleSheet("""
            background-color: #e8f8f5;
            border: 2px solid #1abc9c;
            border-radius: 8px;
            padding: 10px;
            font-size: 12px;
        """)
        return label
    
    def browse_video(self):
        """
        Chọn video file
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn Video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        
        if file_path:
            self.video_path = file_path
            self.video_label.setText(Path(file_path).name)
            self.analyze_btn.setEnabled(True)
            self.statusBar().showMessage(f"Đã chọn: {Path(file_path).name}")
    
    def start_analysis(self):
        """
        Bắt đầu phân tích video
        """
        if not self.video_path:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn video trước!")
            return
        
        # Disable buttons
        self.analyze_btn.setEnabled(False)
        self.browse_btn.setEnabled(False)
        self.save_json_btn.setEnabled(False)
        self.save_pdf_btn.setEnabled(False)
        
        # Show progress
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        
        # Reset results
        self.prediction_label.setText("Đang phân tích...")
        self.prediction_label.setStyleSheet("background-color: #f39c12; border-radius: 8px; padding: 10px;")
        self.artifact_label.setText("Artifact Score\n--")
        self.reality_label.setText("Reality Score\n--")
        self.stress_label.setText("Stress Score\n--")
        self.explanation_text.clear()
        
        # Create worker thread
        run_stress = self.stress_checkbox.isChecked()
        self.worker = AnalysisWorker(self.video_path, self.config, run_stress)
        
        # Connect signals
        self.worker.progress_update.connect(self.on_progress_update)
        self.worker.analysis_complete.connect(self.on_analysis_complete)
        self.worker.analysis_error.connect(self.on_analysis_error)
        
        # Start worker
        self.worker.start()
        self.statusBar().showMessage("Đang phân tích...")
    
    def on_progress_update(self, percentage: int, message: str):
        """
        Cập nhật progress
        """
        self.progress_bar.setValue(percentage)
        self.progress_label.setText(message)
    
    def on_analysis_complete(self, result: dict):
        """
        Xử lý khi phân tích hoàn tất
        """
        self.analysis_result = result
        
        # Hide progress
        self.progress_bar.setVisible(False)
        self.progress_label.setText("")
        
        # Update prediction
        prediction = result['prediction']
        confidence = result['confidence']
        probability = result['final_probability']
        
        pred_text = f"{prediction}\n({confidence} confidence)\nXác suất FAKE: {probability*100:.1f}%"
        
        if prediction == "FAKE":
            bg_color = "#e74c3c"
        else:
            bg_color = "#2ecc71"
        
        self.prediction_label.setText(pred_text)
        self.prediction_label.setStyleSheet(f"""
            background-color: {bg_color};
            color: white;
            border-radius: 8px;
            padding: 10px;
            font-size: 18px;
            font-weight: bold;
        """)
        
        # Update scores
        scores = result['scores']
        self.artifact_label.setText(f"Artifact Score\n{scores['artifact_score']:.3f}")
        self.reality_label.setText(f"Reality Score\n{scores['reality_score']:.3f}")
        self.stress_label.setText(f"Stress Score\n{scores['stress_score']:.3f}")
        
        # Update explanations
        explanations = result.get('explanations', [])
        exp_text = ""
        for i, exp in enumerate(explanations, 1):
            exp_text += f"{i}. {exp}\n\n"
        
        self.explanation_text.setPlainText(exp_text.strip())
        
        # Enable buttons
        self.analyze_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.save_json_btn.setEnabled(True)
        self.save_pdf_btn.setEnabled(True)
        
        self.statusBar().showMessage(f"Phân tích hoàn tất: {prediction}")
        
        QMessageBox.information(
            self,
            "Hoàn tất",
            f"Phân tích hoàn tất!\n\nKết quả: {prediction}\nConfidence: {confidence}"
        )
    
    def on_analysis_error(self, error_msg: str):
        """
        Xử lý lỗi
        """
        self.progress_bar.setVisible(False)
        self.progress_label.setText("")
        
        self.analyze_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        
        self.statusBar().showMessage("Lỗi khi phân tích")
        
        QMessageBox.critical(
            self,
            "Lỗi",
            f"Lỗi khi phân tích video:\n\n{error_msg}"
        )
    
    def save_json(self):
        """
        Lưu kết quả JSON
        """
        if not self.analysis_result:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Lưu JSON",
            "result.json",
            "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.analysis_result, f, indent=2, ensure_ascii=False)
                
                QMessageBox.information(self, "Thành công", f"Đã lưu: {file_path}")
                self.statusBar().showMessage(f"Đã lưu JSON: {Path(file_path).name}")
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", f"Không thể lưu file:\n{e}")
    
    def save_pdf(self):
        """
        Lưu báo cáo PDF
        """
        if not self.analysis_result:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Lưu PDF",
            "report.pdf",
            "PDF Files (*.pdf)"
        )
        
        if file_path:
            try:
                report_gen = ReportGenerator(self.config)
                report_gen.generate_pdf(self.analysis_result, file_path)
                
                QMessageBox.information(self, "Thành công", f"Đã lưu: {file_path}")
                self.statusBar().showMessage(f"Đã lưu PDF: {Path(file_path).name}")
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", f"Không thể tạo PDF:\n{e}")


def main():
    """
    Entry point cho GUI
    """
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Set palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(240, 240, 240))
    palette.setColor(QPalette.WindowText, Qt.black)
    app.setPalette(palette)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()