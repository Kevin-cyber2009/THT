#!/usr/bin/env python3
# client_modern.py
"""
Deepfake Detector Client - Modern ChatGPT-like Interface
Connects to Render.com server only
"""

import sys
import os
from pathlib import Path
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
import requests
from datetime import datetime
import json

# Get base path
if getattr(sys, 'frozen', False):
    BASE_PATH = os.path.dirname(sys.executable)
else:
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))

os.chdir(BASE_PATH)

# HARDCODED RENDER URL - Change this to your Render URL
RENDER_API_URL = "https://your-app.onrender.com"  # ← CHANGE THIS!


class AnalysisThread(QThread):
    """Analysis thread"""
    progress = Signal(str)
    finished = Signal(dict)
    error = Signal(str)
    
    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
    
    def run(self):
        try:
            self.progress.emit("Checking server...")
            
            # Check health
            try:
                health = requests.get(f"{RENDER_API_URL}/health", timeout=10)
                if health.status_code != 200:
                    self.error.emit("Server offline. Please try again later.")
                    return
            except:
                self.error.emit("Cannot connect to server. Please check your internet connection.")
                return
            
            self.progress.emit("Uploading video...")
            
            # Upload and analyze
            with open(self.video_path, 'rb') as f:
                files = {'video': f}
                self.progress.emit("Analyzing...")
                response = requests.post(
                    f"{RENDER_API_URL}/api/analyze",
                    files=files,
                    timeout=300
                )
            
            if response.status_code != 200:
                self.error.emit(f"Analysis failed: {response.status_code}")
                return
            
            result = response.json()
            
            if not result.get('success'):
                self.error.emit(result.get('error', 'Unknown error'))
                return
            
            result['video_path'] = self.video_path
            result['timestamp'] = datetime.now().isoformat()
            
            self.progress.emit("Complete!")
            self.finished.emit(result)
            
        except Exception as e:
            self.error.emit(f"Error: {str(e)}")


class ModernWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deepfake Detector")
        self.setMinimumSize(900, 650)
        
        # History
        self.history_file = os.path.join(BASE_PATH, "history.json")
        self.history = self.load_history()
        
        self.setup_ui()
        self.apply_modern_style()
    
    def setup_ui(self):
        """Setup modern UI"""
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        # Main layout
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Top bar
        top_bar = self.create_top_bar()
        main_layout.addWidget(top_bar)
        
        # Content area
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(40, 40, 40, 40)
        content_layout.setSpacing(30)
        
        # Welcome message
        welcome = QLabel("Deepfake Video Detector")
        welcome.setObjectName("welcome")
        welcome.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(welcome)
        
        subtitle = QLabel("Upload a video to check if it's real or fake")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(subtitle)
        
        content_layout.addSpacing(20)
        
        # Upload area
        upload_area = self.create_upload_area()
        content_layout.addWidget(upload_area)
        
        content_layout.addSpacing(20)
        
        # Progress area
        self.progress_widget = self.create_progress_area()
        self.progress_widget.setVisible(False)
        content_layout.addWidget(self.progress_widget)
        
        # Results area
        self.results_widget = self.create_results_area()
        self.results_widget.setVisible(False)
        content_layout.addWidget(self.results_widget)
        
        content_layout.addStretch()
        
        main_layout.addWidget(content)
    
    def create_top_bar(self):
        """Create top navigation bar"""
        bar = QWidget()
        bar.setObjectName("topBar")
        bar.setFixedHeight(60)
        
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(20, 0, 20, 0)
        
        # Logo/Title
        title = QLabel("🔍 Deepfake Detector")
        title.setObjectName("appTitle")
        layout.addWidget(title)
        
        layout.addStretch()
        
        # History button
        history_btn = QPushButton("📜 History")
        history_btn.setObjectName("topBarButton")
        history_btn.clicked.connect(self.show_history)
        layout.addWidget(history_btn)
        
        # About button
        about_btn = QPushButton("ℹ️ About")
        about_btn.setObjectName("topBarButton")
        about_btn.clicked.connect(self.show_about)
        layout.addWidget(about_btn)
        
        return bar
    
    def create_upload_area(self):
        """Create upload area"""
        area = QWidget()
        area.setObjectName("uploadArea")
        
        layout = QVBoxLayout(area)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)
        
        # Icon
        icon_label = QLabel("📹")
        icon_label.setObjectName("uploadIcon")
        icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon_label)
        
        # Text
        text = QLabel("Click to upload or drag and drop")
        text.setObjectName("uploadText")
        text.setAlignment(Qt.AlignCenter)
        layout.addWidget(text)
        
        subtext = QLabel("MP4, AVI, MOV, MKV (max 100MB)")
        subtext.setObjectName("uploadSubtext")
        subtext.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtext)
        
        # Selected file label
        self.selected_file_label = QLabel("")
        self.selected_file_label.setObjectName("selectedFile")
        self.selected_file_label.setAlignment(Qt.AlignCenter)
        self.selected_file_label.setWordWrap(True)
        layout.addWidget(self.selected_file_label)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(15)
        
        # Browse button
        browse_btn = QPushButton("Choose File")
        browse_btn.setObjectName("primaryButton")
        browse_btn.clicked.connect(self.browse_file)
        btn_layout.addWidget(browse_btn)
        
        # Analyze button
        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.setObjectName("analyzeButton")
        self.analyze_btn.clicked.connect(self.analyze_video)
        self.analyze_btn.setEnabled(False)
        btn_layout.addWidget(analyze_btn)
        
        layout.addLayout(btn_layout)
        
        # Make area clickable
        area.mousePressEvent = lambda e: self.browse_file()
        
        return area
    
    def create_progress_area(self):
        """Create progress area"""
        widget = QWidget()
        widget.setObjectName("progressArea")
        
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(15)
        
        self.progress_label = QLabel("Processing...")
        self.progress_label.setObjectName("progressLabel")
        self.progress_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("modernProgress")
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)
        
        return widget
    
    def create_results_area(self):
        """Create results area"""
        widget = QWidget()
        widget.setObjectName("resultsArea")
        
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)
        
        # Result card
        self.result_card = QWidget()
        self.result_card.setObjectName("resultCard")
        card_layout = QVBoxLayout(self.result_card)
        card_layout.setContentsMargins(30, 30, 30, 30)
        card_layout.setSpacing(20)
        
        # Result text browser
        self.result_display = QTextEdit()
        self.result_display.setObjectName("resultDisplay")
        self.result_display.setReadOnly(True)
        card_layout.addWidget(self.result_display)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        new_btn = QPushButton("Analyze Another")
        new_btn.setObjectName("secondaryButton")
        new_btn.clicked.connect(self.reset_ui)
        btn_layout.addWidget(new_btn)
        
        btn_layout.addStretch()
        
        card_layout.addLayout(btn_layout)
        
        layout.addWidget(self.result_card)
        
        return widget
    
    def apply_modern_style(self):
        """Apply modern ChatGPT-like styling"""
        self.setStyleSheet("""
            /* Main Window */
            QMainWindow {
                background-color: #ffffff;
            }
            
            /* Top Bar */
            #topBar {
                background-color: #f7f7f8;
                border-bottom: 1px solid #e5e5e5;
            }
            
            #appTitle {
                font-size: 18px;
                font-weight: 600;
                color: #202123;
            }
            
            #topBarButton {
                background-color: transparent;
                border: none;
                color: #565869;
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 14px;
            }
            
            #topBarButton:hover {
                background-color: #ececf1;
            }
            
            /* Welcome Text */
            #welcome {
                font-size: 32px;
                font-weight: 600;
                color: #202123;
            }
            
            #subtitle {
                font-size: 16px;
                color: #565869;
            }
            
            /* Upload Area */
            #uploadArea {
                background-color: #ffffff;
                border: 2px dashed #d1d5db;
                border-radius: 12px;
                min-height: 280px;
            }
            
            #uploadArea:hover {
                border-color: #10a37f;
                background-color: #f9fafb;
            }
            
            #uploadIcon {
                font-size: 64px;
            }
            
            #uploadText {
                font-size: 16px;
                font-weight: 500;
                color: #202123;
            }
            
            #uploadSubtext {
                font-size: 14px;
                color: #6e6e80;
            }
            
            #selectedFile {
                font-size: 14px;
                color: #10a37f;
                font-weight: 500;
            }
            
            /* Buttons */
            #primaryButton {
                background-color: #10a37f;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-size: 14px;
                font-weight: 500;
                min-width: 140px;
            }
            
            #primaryButton:hover {
                background-color: #0d8f6f;
            }
            
            #primaryButton:pressed {
                background-color: #0b7a5f;
            }
            
            #analyzeButton {
                background-color: #2563eb;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-size: 14px;
                font-weight: 500;
                min-width: 140px;
            }
            
            #analyzeButton:hover:enabled {
                background-color: #1d4ed8;
            }
            
            #analyzeButton:disabled {
                background-color: #9ca3af;
            }
            
            #secondaryButton {
                background-color: #f7f7f8;
                color: #202123;
                border: 1px solid #d1d5db;
                padding: 10px 20px;
                border-radius: 6px;
                font-size: 14px;
                font-weight: 500;
            }
            
            #secondaryButton:hover {
                background-color: #ececf1;
            }
            
            /* Progress Area */
            #progressArea {
                background-color: #f9fafb;
                border-radius: 12px;
                border: 1px solid #e5e5e5;
            }
            
            #progressLabel {
                font-size: 16px;
                color: #202123;
                font-weight: 500;
            }
            
            #modernProgress {
                border: none;
                background-color: #e5e7eb;
                border-radius: 4px;
                height: 8px;
            }
            
            #modernProgress::chunk {
                background-color: #10a37f;
                border-radius: 4px;
            }
            
            /* Results Area */
            #resultCard {
                background-color: #ffffff;
                border: 1px solid #e5e5e5;
                border-radius: 12px;
            }
            
            #resultDisplay {
                background-color: #ffffff;
                border: none;
                font-size: 14px;
                color: #202123;
            }
        """)
    
    def browse_file(self):
        """Browse for video file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)"
        )
        
        if file_path:
            self.video_path = file_path
            filename = Path(file_path).name
            self.selected_file_label.setText(f"✓ Selected: {filename}")
            self.analyze_btn.setEnabled(True)
    
    def analyze_video(self):
        """Start analysis"""
        if not hasattr(self, 'video_path'):
            return
        
        # Hide upload, show progress
        self.results_widget.setVisible(False)
        self.progress_widget.setVisible(True)
        self.analyze_btn.setEnabled(False)
        
        # Start thread
        self.thread = AnalysisThread(self.video_path)
        self.thread.progress.connect(self.update_progress)
        self.thread.finished.connect(self.show_results)
        self.thread.error.connect(self.show_error)
        self.thread.start()
    
    def update_progress(self, message):
        """Update progress"""
        self.progress_label.setText(message)
    
    def show_results(self, result):
        """Display results"""
        # Hide progress, show results
        self.progress_widget.setVisible(False)
        self.results_widget.setVisible(True)
        self.analyze_btn.setEnabled(True)
        
        # Save to history
        self.history.append(result)
        self.save_history()
        
        # Display
        prediction = result.get('prediction', 'UNKNOWN')
        prob_fake = result.get('probability_fake', 0) * 100
        prob_real = result.get('probability_real', 0) * 100
        confidence = result.get('confidence', 'UNKNOWN')
        
        if prediction == 'FAKE':
            color = '#dc2626'
            emoji = '🚨'
            verdict = 'FAKE'
        else:
            color = '#16a34a'
            emoji = '✅'
            verdict = 'REAL'
        
        html = f"""
        <div style="text-align: center; padding: 30px 20px; background: {color}; color: white; border-radius: 10px; margin-bottom: 25px;">
            <div style="font-size: 48px; margin-bottom: 10px;">{emoji}</div>
            <div style="font-size: 28px; font-weight: 600; margin-bottom: 8px;">{verdict}</div>
            <div style="font-size: 16px; opacity: 0.9;">Confidence: {confidence}</div>
        </div>
        
        <div style="background: #f9fafb; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
            <div style="font-size: 16px; font-weight: 600; color: #202123; margin-bottom: 15px;">Probability Breakdown</div>
            <div style="margin-bottom: 10px;">
                <div style="color: #6e6e80; font-size: 14px; margin-bottom: 5px;">Fake: {prob_fake:.1f}%</div>
                <div style="background: #e5e7eb; height: 8px; border-radius: 4px; overflow: hidden;">
                    <div style="background: #dc2626; height: 100%; width: {prob_fake}%;"></div>
                </div>
            </div>
            <div>
                <div style="color: #6e6e80; font-size: 14px; margin-bottom: 5px;">Real: {prob_real:.1f}%</div>
                <div style="background: #e5e7eb; height: 8px; border-radius: 4px; overflow: hidden;">
                    <div style="background: #16a34a; height: 100%; width: {prob_real}%;"></div>
                </div>
            </div>
        </div>
        
        <div style="background: #f9fafb; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
            <div style="font-size: 16px; font-weight: 600; color: #202123; margin-bottom: 12px;">Analysis Details</div>
            <div style="color: #565869; font-size: 14px; line-height: 1.6;">
                <div style="margin-bottom: 8px;"><strong>Artifact Score:</strong> {result.get('artifact_score', 0):.3f}</div>
                <div><strong>Reality Score:</strong> {result.get('reality_score', 0):.3f}</div>
            </div>
        </div>
        
        <div style="background: #f9fafb; padding: 20px; border-radius: 8px;">
            <div style="font-size: 16px; font-weight: 600; color: #202123; margin-bottom: 12px;">Key Findings</div>
            <div style="color: #565869; font-size: 14px; line-height: 1.8;">
        """
        
        for i, explanation in enumerate(result.get('explanations', []), 1):
            html += f"<div style='margin-bottom: 8px;'>• {explanation}</div>"
        
        html += "</div></div>"
        
        self.result_display.setHtml(html)
    
    def show_error(self, error):
        """Show error"""
        self.progress_widget.setVisible(False)
        self.analyze_btn.setEnabled(True)
        
        QMessageBox.critical(self, "Error", error)
    
    def reset_ui(self):
        """Reset to initial state"""
        self.results_widget.setVisible(False)
        self.selected_file_label.setText("")
        self.analyze_btn.setEnabled(False)
        if hasattr(self, 'video_path'):
            delattr(self, 'video_path')
    
    def show_history(self):
        """Show history dialog"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Analysis History")
        dialog.setMinimumSize(600, 400)
        
        layout = QVBoxLayout(dialog)
        
        list_widget = QListWidget()
        for item in reversed(self.history[-20:]):
            timestamp = item.get('timestamp', '')[:19]
            prediction = item.get('prediction', 'UNKNOWN')
            video_name = Path(item.get('video_path', 'Unknown')).name
            
            emoji = '🚨' if prediction == 'FAKE' else '✅'
            list_widget.addItem(f"{emoji} {timestamp} - {video_name} → {prediction}")
        
        layout.addWidget(list_widget)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        dialog.exec()
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.information(
            self,
            "About",
            f"<h3>Deepfake Detector</h3>"
            f"<p>Version 1.0.0</p>"
            f"<p>Server: {RENDER_API_URL}</p>"
            f"<p>Powered by AI & Machine Learning</p>"
        )
    
    def load_history(self):
        """Load history"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def save_history(self):
        """Save history"""
        try:
            os.makedirs(os.path.dirname(self.history_file) if os.path.dirname(self.history_file) else '.', exist_ok=True)
            self.history = self.history[-50:]
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except:
            pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Set app info
    app.setApplicationName("Deepfake Detector")
    app.setOrganizationName("AI Research")
    
    window = ModernWindow()
    window.show()
    
    sys.exit(app.exec())