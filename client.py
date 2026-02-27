#!/usr/bin/env python3
# client.py
"""
Deepfake Detector Client - Connects to Server API
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


class AnalysisThread(QThread):
    """Thread để gọi API (không block UI)"""
    progress = Signal(str)
    finished = Signal(dict)
    error = Signal(str)
    
    def __init__(self, video_path, api_url):
        super().__init__()
        self.video_path = video_path
        self.api_url = api_url
    
    def run(self):
        try:
            self.progress.emit("🔍 Đang kết nối server...")
            
            # Check server health
            try:
                health_response = requests.get(
                    f"{self.api_url}/health",
                    timeout=10
                )
                
                if health_response.status_code != 200:
                    self.error.emit("❌ Server không phản hồi")
                    return
            except Exception as e:
                self.error.emit(f"❌ Không thể kết nối server:\n{e}\n\nKiểm tra:\n1. Server đang chạy?\n2. API URL đúng?")
                return
            
            self.progress.emit("📤 Đang upload video...")
            
            # Upload video to API
            with open(self.video_path, 'rb') as f:
                files = {'video': f}
                
                self.progress.emit("⚙️ Đang phân tích trên server...")
                
                response = requests.post(
                    f"{self.api_url}/api/analyze",
                    files=files,
                    timeout=300  # 5 minutes
                )
            
            if response.status_code != 200:
                self.error.emit(f"❌ Lỗi từ server: {response.status_code}\n{response.text}")
                return
            
            result = response.json()
            
            if not result.get('success', False):
                self.error.emit(f"❌ Phân tích thất bại: {result.get('error', 'Unknown error')}")
                return
            
            # Add metadata
            result['video_path'] = self.video_path
            result['timestamp'] = datetime.now().isoformat()
            
            self.progress.emit("✅ Hoàn tất!")
            self.finished.emit(result)
            
        except requests.exceptions.Timeout:
            self.error.emit("⏱️ Timeout: Video quá lớn hoặc server quá chậm")
        except requests.exceptions.ConnectionError:
            self.error.emit("🔌 Lỗi kết nối: Không thể kết nối đến server")
        except Exception as e:
            import traceback
            self.error.emit(f"❌ Lỗi: {str(e)}\n\n{traceback.format_exc()}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔍 Deepfake Detector - Client")
        self.setMinimumSize(900, 700)
        
        # API Configuration
        self.config_file = os.path.join(BASE_PATH, "client_config.json")
        self.api_url = self.load_api_url()
        
        # History
        self.history_file = os.path.join(BASE_PATH, "history.json")
        self.history = self.load_history()
        
        self.setup_ui()
        self.apply_styles()
    
    def load_api_url(self):
        """Load API URL from config"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    return config.get('api_url', 'http://localhost:5000')
            except:
                pass
        return 'http://localhost:5000'
    
    def save_api_url(self):
        """Save API URL to config"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump({'api_url': self.api_url}, f)
        except:
            pass
    
    def setup_ui(self):
        """Setup UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # Header
        header = QLabel("🔍 Deepfake Video Detector")
        header.setStyleSheet("font-size: 28px; font-weight: bold; color: #1976D2;")
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        # API Status
        self.api_status_label = QLabel(f"📡 Server: {self.api_url}")
        self.api_status_label.setStyleSheet("font-size: 12px; color: #666;")
        self.api_status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.api_status_label)
        
        # Tabs
        self.tabs = QTabWidget()
        
        # Tab 1: Analyze
        analyze_tab = self.create_analyze_tab()
        self.tabs.addTab(analyze_tab, "🔍 Phân Tích")
        
        # Tab 2: Settings
        settings_tab = self.create_settings_tab()
        self.tabs.addTab(settings_tab, "⚙️ Settings")
        
        # Tab 3: History
        history_tab = self.create_history_tab()
        self.tabs.addTab(history_tab, "📜 Lịch Sử")
        
        layout.addWidget(self.tabs)
    
    def create_analyze_tab(self):
        """Create analyze tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Input group
        input_group = QGroupBox("📁 Chọn Video")
        input_layout = QVBoxLayout()
        
        file_layout = QHBoxLayout()
        self.file_input = QLineEdit()
        self.file_input.setPlaceholderText("Chọn file video để phân tích...")
        browse_btn = QPushButton("📂 Browse")
        browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(self.file_input, 3)
        file_layout.addWidget(browse_btn, 1)
        input_layout.addLayout(file_layout)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Analyze button
        self.analyze_btn = QPushButton("🔍 Phân Tích Video")
        self.analyze_btn.setMinimumHeight(50)
        self.analyze_btn.clicked.connect(self.analyze_video)
        layout.addWidget(self.analyze_btn)
        
        # Progress
        self.progress_label = QLabel("")
        self.progress_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Results
        results_group = QGroupBox("📊 Kết Quả")
        results_layout = QVBoxLayout()
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMinimumHeight(300)
        results_layout.addWidget(self.result_text)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        layout.addStretch()
        return tab
    
    def create_settings_tab(self):
        """Create settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # API Settings
        api_group = QGroupBox("🌐 API Configuration")
        api_layout = QVBoxLayout()
        
        # API URL
        url_layout = QHBoxLayout()
        url_layout.addWidget(QLabel("API URL:"))
        self.api_url_input = QLineEdit(self.api_url)
        url_layout.addWidget(self.api_url_input)
        
        save_btn = QPushButton("💾 Save")
        save_btn.clicked.connect(self.save_api_settings)
        url_layout.addWidget(save_btn)
        
        api_layout.addLayout(url_layout)
        
        # Test connection
        test_btn = QPushButton("🔌 Test Connection")
        test_btn.clicked.connect(self.test_connection)
        api_layout.addWidget(test_btn)
        
        self.connection_status = QLabel("")
        api_layout.addWidget(self.connection_status)
        
        api_group.setLayout(api_layout)
        layout.addWidget(api_group)
        
        # Common URLs
        common_group = QGroupBox("🔗 Common Configurations")
        common_layout = QVBoxLayout()
        
        common_urls = [
            ("Local Server", "http://localhost:5000"),
            ("Render.com", "https://your-app.onrender.com"),
            ("Railway.app", "https://your-app.railway.app"),
            ("Custom", "https://your-domain.com")
        ]
        
        for name, url in common_urls:
            btn = QPushButton(f"{name}: {url}")
            btn.clicked.connect(lambda checked, u=url: self.set_api_url(u))
            common_layout.addWidget(btn)
        
        common_group.setLayout(common_layout)
        layout.addWidget(common_group)
        
        # Info
        info_group = QGroupBox("ℹ️ Information")
        info_layout = QVBoxLayout()
        
        info_text = QLabel(
            "Client Version: 1.0.0\n"
            "Author: Your Name\n"
            "Description: Client app to connect to Deepfake Detector API"
        )
        info_layout.addWidget(info_text)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        layout.addStretch()
        return tab
    
    def create_history_tab(self):
        """Create history tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        self.history_list = QListWidget()
        self.history_list.itemClicked.connect(self.show_history_item)
        layout.addWidget(self.history_list)
        
        clear_btn = QPushButton("🗑️ Xóa Lịch Sử")
        clear_btn.clicked.connect(self.clear_history)
        layout.addWidget(clear_btn)
        
        self.update_history_list()
        return tab
    
    def apply_styles(self):
        """Apply custom styles"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #ddd;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QPushButton {
                background-color: #1976D2;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1565C0;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
            QLineEdit {
                padding: 8px;
                border: 2px solid #ddd;
                border-radius: 5px;
            }
            QTextEdit {
                border: 2px solid #ddd;
                border-radius: 5px;
                padding: 10px;
            }
        """)
    
    def browse_file(self):
        """Browse for video file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)"
        )
        
        if file_path:
            self.file_input.setText(file_path)
    
    def save_api_settings(self):
        """Save API settings"""
        new_url = self.api_url_input.text().strip()
        
        if not new_url:
            QMessageBox.warning(self, "Warning", "API URL không được để trống")
            return
        
        self.api_url = new_url
        self.save_api_url()
        self.api_status_label.setText(f"📡 Server: {self.api_url}")
        
        QMessageBox.information(self, "Success", "✅ API URL đã được lưu!")
    
    def set_api_url(self, url):
        """Set API URL from preset"""
        self.api_url_input.setText(url)
    
    def test_connection(self):
        """Test connection to server"""
        try:
            self.connection_status.setText("🔄 Testing...")
            self.connection_status.setStyleSheet("color: orange;")
            QApplication.processEvents()
            
            response = requests.get(f"{self.api_url}/health", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                status = "✅ Connected!\n"
                if data.get('model_loaded'):
                    status += "✅ Model loaded on server\n"
                status += f"⏱️ Server uptime: {data.get('uptime_seconds', 0):.1f}s"
                self.connection_status.setStyleSheet("color: green;")
            else:
                status = f"❌ Server error: {response.status_code}"
                self.connection_status.setStyleSheet("color: red;")
            
            self.connection_status.setText(status)
            
        except Exception as e:
            self.connection_status.setText(f"❌ Connection failed:\n{e}")
            self.connection_status.setStyleSheet("color: red;")
    
    def analyze_video(self):
        """Start video analysis"""
        video_path = self.file_input.text().strip()
        
        if not video_path:
            QMessageBox.warning(self, "Warning", "⚠️ Vui lòng chọn video")
            return
        
        if not os.path.exists(video_path):
            QMessageBox.warning(self, "Warning", "⚠️ File không tồn tại")
            return
        
        # Disable UI
        self.analyze_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.result_text.clear()
        
        # Start analysis thread
        self.thread = AnalysisThread(video_path, self.api_url)
        self.thread.progress.connect(self.update_progress)
        self.thread.finished.connect(self.analysis_complete)
        self.thread.error.connect(self.analysis_error)
        self.thread.start()
    
    def update_progress(self, message):
        """Update progress message"""
        self.progress_label.setText(message)
    
    def analysis_complete(self, result):
        """Handle analysis completion"""
        self.analyze_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_label.setText("")
        
        # Display result
        self.display_result(result)
        
        # Save to history
        self.history.append(result)
        self.save_history()
        self.update_history_list()
    
    def analysis_error(self, error_msg):
        """Handle analysis error"""
        self.analyze_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_label.setText("")
        
        QMessageBox.critical(self, "Error", error_msg)
    
    def display_result(self, result):
        """Display analysis result"""
        prediction = result.get('prediction', 'UNKNOWN')
        prob_fake = result.get('probability_fake', 0) * 100
        prob_real = result.get('probability_real', 0) * 100
        confidence = result.get('confidence', 'UNKNOWN')
        
        # Color based on prediction
        if prediction == 'FAKE':
            color = '#D32F2F'
            emoji = '🚨'
        else:
            color = '#388E3C'
            emoji = '✅'
        
        html = f"""
        <div style="background: {color}; color: white; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 36px;">{emoji} {prediction}</h1>
            <p style="margin: 10px 0 0 0; font-size: 18px;">Độ tin cậy: {confidence}</p>
        </div>
        
        <div style="background: #E3F2FD; padding: 15px; border-radius: 5px; margin: 10px 0;">
            <h3 style="color: #1976D2;">📊 Chi tiết xác suất:</h3>
            <p><strong>FAKE:</strong> {prob_fake:.1f}%</p>
            <p><strong>REAL:</strong> {prob_real:.1f}%</p>
        </div>
        
        <div style="background: #FFF3E0; padding: 15px; border-radius: 5px; margin: 10px 0;">
            <h3 style="color: #F57C00;">🔍 Phân tích chi tiết:</h3>
            <p><strong>Artifact Score:</strong> {result.get('artifact_score', 0):.3f}</p>
            <p><strong>Reality Score:</strong> {result.get('reality_score', 0):.3f}</p>
        </div>
        
        <div style="background: #E8F5E9; padding: 15px; border-radius: 5px; margin: 10px 0;">
            <h3 style="color: #388E3C;">💡 Giải thích:</h3>
        """
        
        for i, explanation in enumerate(result.get('explanations', []), 1):
            html += f"<p>{i}. {explanation}</p>"
        
        html += "</div>"
        
        # Metadata
        metadata = result.get('metadata', {})
        if metadata:
            html += """
            <div style="background: #F3E5F5; padding: 15px; border-radius: 5px; margin: 10px 0;">
                <h3 style="color: #7B1FA2;">📹 Thông tin video:</h3>
            """
            
            if 'num_frames' in metadata:
                html += f"<p><strong>Số frames:</strong> {metadata['num_frames']}</p>"
            if 'duration' in metadata:
                html += f"<p><strong>Thời lượng:</strong> {metadata['duration']:.1f}s</p>"
            if 'fps' in metadata:
                html += f"<p><strong>FPS:</strong> {metadata['fps']:.1f}</p>"
            
            html += "</div>"
        
        self.result_text.setHtml(html)
    
    def load_history(self):
        """Load history"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def save_history(self):
        """Save history"""
        try:
            os.makedirs(os.path.dirname(self.history_file) if os.path.dirname(self.history_file) else '.', exist_ok=True)
            self.history = self.history[-50:]  # Keep last 50
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        except:
            pass
    
    def update_history_list(self):
        """Update history list"""
        self.history_list.clear()
        
        for item in reversed(self.history):
            timestamp = item.get('timestamp', 'Unknown')
            prediction = item.get('prediction', 'UNKNOWN')
            video_name = Path(item.get('video_path', 'Unknown')).name
            
            emoji = '🚨' if prediction == 'FAKE' else '✅'
            
            self.history_list.addItem(f"{emoji} {timestamp[:19]} - {video_name} → {prediction}")
    
    def show_history_item(self, item):
        """Show history item details"""
        index = self.history_list.row(item)
        result = self.history[-(index + 1)]
        
        self.tabs.setCurrentIndex(0)
        self.display_result(result)
    
    def clear_history(self):
        """Clear history"""
        reply = QMessageBox.question(
            self,
            "Confirm",
            "Bạn có chắc muốn xóa lịch sử?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.history = []
            self.save_history()
            self.update_history_list()
            self.result_text.clear()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Set app icon if exists
    icon_path = os.path.join(BASE_PATH, "icon.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())
