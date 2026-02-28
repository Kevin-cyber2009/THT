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

if getattr(sys, 'frozen', False):
    BASE_PATH = os.path.dirname(sys.executable)
else:
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))

os.chdir(BASE_PATH)

RENDER_API_URL = "https://your-app.onrender.com"  # ← CHANGE THIS!
TEMP_DIR = os.path.join(BASE_PATH, "temp_downloads")


# ─────────────────────────────────────────────
# THREADS
# ─────────────────────────────────────────────

class DownloadThread(QThread):
    """Download video từ YouTube / Facebook bằng yt-dlp"""
    progress = Signal(str)
    finished = Signal(str)
    error    = Signal(str)

    def __init__(self, url: str, output_dir: str):
        super().__init__()
        self.url = url
        self.output_dir = output_dir

    def run(self):
        try:
            import subprocess

            self.progress.emit("Đang kiểm tra yt-dlp...")
            check = subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True)
            if check.returncode != 0:
                self.error.emit("yt-dlp chưa được cài.\nChạy: pip install yt-dlp")
                return

            os.makedirs(self.output_dir, exist_ok=True)
            output_tpl = os.path.join(self.output_dir, "downloaded_%(id)s.%(ext)s")

            self.progress.emit("Đang download video...")
            cmd = [
                "yt-dlp",
                "-f", "best[height<=1080][ext=mp4]/best[height<=1080]/best",
                "--max-filesize", "200M",
                "--no-playlist",
                "-o", output_tpl,
                self.url,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                raise Exception(result.stderr or "Download thất bại")

            files = sorted(
                Path(self.output_dir).glob("downloaded_*"),
                key=lambda f: f.stat().st_mtime, reverse=True
            )
            if not files:
                raise Exception("Không tìm thấy file đã download")

            self.progress.emit("Download hoàn tất!")
            self.finished.emit(str(files[0]))

        except Exception as e:
            self.error.emit(str(e))


class AnalysisThread(QThread):
    """Gửi video lên server phân tích"""
    progress = Signal(str)
    finished = Signal(dict)
    error    = Signal(str)

    def __init__(self, video_path: str):
        super().__init__()
        self.video_path = video_path

    def run(self):
        try:
            self.progress.emit("Đang kiểm tra server...")
            try:
                health = requests.get(f"{RENDER_API_URL}/health", timeout=15)
                if health.status_code != 200:
                    self.error.emit("Server offline. Vui lòng thử lại sau.")
                    return
            except Exception:
                self.error.emit("Không kết nối được server. Kiểm tra internet.")
                return

            self.progress.emit("Đang upload video...")
            with open(self.video_path, "rb") as f:
                self.progress.emit("Đang phân tích AI...")
                response = requests.post(
                    f"{RENDER_API_URL}/api/analyze",
                    files={"video": f},
                    timeout=300,
                )

            if response.status_code != 200:
                self.error.emit(f"Server lỗi: {response.status_code}")
                return

            result = response.json()
            if not result.get("success"):
                self.error.emit(result.get("error", "Lỗi không xác định"))
                return

            result["video_path"] = self.video_path
            result["timestamp"]  = datetime.now().isoformat()
            self.progress.emit("Hoàn tất!")
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(f"Lỗi: {str(e)}")


# ─────────────────────────────────────────────
# MAIN WINDOW
# ─────────────────────────────────────────────

class ModernWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deepfake Detector")
        self.setMinimumSize(960, 700)

        self.history_file = os.path.join(BASE_PATH, "history.json")
        self.history = self.load_history()
        self._input_mode = "file"
        self._video_path = None

        self.setup_ui()
        self.apply_modern_style()

    # ── UI BUILD ──────────────────────────────

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self.create_top_bar())

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        wrapper = QWidget()
        wlayout = QVBoxLayout(wrapper)
        wlayout.setContentsMargins(40, 40, 40, 40)
        wlayout.setSpacing(24)

        welcome = QLabel("Deepfake Video Detector")
        welcome.setObjectName("welcome")
        welcome.setAlignment(Qt.AlignCenter)
        wlayout.addWidget(welcome)

        subtitle = QLabel("Upload file hoặc paste link YouTube / Facebook")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        wlayout.addWidget(subtitle)

        wlayout.addWidget(self.create_mode_toggle())

        self.stack = QStackedWidget()
        self.stack.addWidget(self.create_file_panel())
        self.stack.addWidget(self.create_url_panel())
        wlayout.addWidget(self.stack)

        self.progress_widget = self.create_progress_area()
        self.progress_widget.setVisible(False)
        wlayout.addWidget(self.progress_widget)

        self.results_widget = self.create_results_area()
        self.results_widget.setVisible(False)
        wlayout.addWidget(self.results_widget)

        wlayout.addStretch()
        scroll.setWidget(wrapper)
        root.addWidget(scroll)

    def create_top_bar(self):
        bar = QWidget()
        bar.setObjectName("topBar")
        bar.setFixedHeight(60)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(20, 0, 20, 0)

        title = QLabel("🔍 Deepfake Detector")
        title.setObjectName("appTitle")
        layout.addWidget(title)
        layout.addStretch()

        for label, slot in [("📜 History", self.show_history), ("ℹ️ About", self.show_about)]:
            btn = QPushButton(label)
            btn.setObjectName("topBarButton")
            btn.clicked.connect(slot)
            layout.addWidget(btn)
        return bar

    def create_mode_toggle(self):
        wrapper = QWidget()
        layout = QHBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.setAlignment(Qt.AlignCenter)

        self.btn_file_mode = QPushButton("📂  File trên máy")
        self.btn_file_mode.setObjectName("modeActive")
        self.btn_file_mode.setCheckable(True)
        self.btn_file_mode.setChecked(True)
        self.btn_file_mode.clicked.connect(lambda: self.switch_mode("file"))

        self.btn_url_mode = QPushButton("🔗  Link YouTube / Facebook")
        self.btn_url_mode.setObjectName("modeInactive")
        self.btn_url_mode.setCheckable(True)
        self.btn_url_mode.clicked.connect(lambda: self.switch_mode("url"))

        layout.addWidget(self.btn_file_mode)
        layout.addWidget(self.btn_url_mode)
        return wrapper

    def create_file_panel(self):
        area = QWidget()
        area.setObjectName("uploadArea")
        layout = QVBoxLayout(area)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(16)

        for txt, obj in [("📹", "uploadIcon"), ("Kéo thả hoặc nhấn chọn file", "uploadText"),
                         ("MP4, AVI, MOV, MKV (tối đa 200MB)", "uploadSubtext")]:
            lbl = QLabel(txt)
            lbl.setObjectName(obj)
            lbl.setAlignment(Qt.AlignCenter)
            layout.addWidget(lbl)

        self.file_selected_label = QLabel("")
        self.file_selected_label.setObjectName("selectedFile")
        self.file_selected_label.setAlignment(Qt.AlignCenter)
        self.file_selected_label.setWordWrap(True)
        layout.addWidget(self.file_selected_label)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        btn_row.setAlignment(Qt.AlignCenter)

        browse_btn = QPushButton("Chọn File")
        browse_btn.setObjectName("primaryButton")
        browse_btn.clicked.connect(self.browse_file)
        btn_row.addWidget(browse_btn)

        self.analyze_file_btn = QPushButton("Phân tích")
        self.analyze_file_btn.setObjectName("analyzeButton")
        self.analyze_file_btn.clicked.connect(self.start_file_analysis)
        self.analyze_file_btn.setEnabled(False)
        btn_row.addWidget(self.analyze_file_btn)

        layout.addLayout(btn_row)
        area.mousePressEvent = lambda e: self.browse_file()
        return area

    def create_url_panel(self):
        panel = QWidget()
        panel.setObjectName("urlPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(16)

        for txt, obj in [("🔗", "uploadIcon"),
                         ("Dán link YouTube hoặc Facebook vào ô bên dưới", "uploadText"),
                         ("Video sẽ được tải về tự động trước khi phân tích", "uploadSubtext")]:
            lbl = QLabel(txt)
            lbl.setObjectName(obj)
            lbl.setAlignment(Qt.AlignCenter)
            layout.addWidget(lbl)

        url_row = QHBoxLayout()
        url_row.setSpacing(8)

        self.url_input = QLineEdit()
        self.url_input.setObjectName("urlInput")
        self.url_input.setPlaceholderText(
            "https://www.youtube.com/watch?v=...  hoặc  https://www.facebook.com/..."
        )
        self.url_input.textChanged.connect(self.on_url_changed)
        self.url_input.returnPressed.connect(self.start_url_analysis)
        url_row.addWidget(self.url_input)

        paste_btn = QPushButton("📋 Dán")
        paste_btn.setObjectName("pasteButton")
        paste_btn.clicked.connect(self.paste_url)
        url_row.addWidget(paste_btn)

        layout.addLayout(url_row)

        self.url_status_label = QLabel("")
        self.url_status_label.setObjectName("urlStatus")
        self.url_status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.url_status_label)

        self.analyze_url_btn = QPushButton("⬇️  Download & Phân tích")
        self.analyze_url_btn.setObjectName("analyzeButton")
        self.analyze_url_btn.clicked.connect(self.start_url_analysis)
        self.analyze_url_btn.setEnabled(False)
        layout.addWidget(self.analyze_url_btn)

        platforms = QLabel("✅ Hỗ trợ: YouTube · Facebook · TikTok · Instagram · Twitter/X · và 1000+ nguồn khác")
        platforms.setObjectName("platformHint")
        platforms.setAlignment(Qt.AlignCenter)
        platforms.setWordWrap(True)
        layout.addWidget(platforms)

        return panel

    def create_progress_area(self):
        widget = QWidget()
        widget.setObjectName("progressArea")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(12)

        self.progress_label = QLabel("Đang xử lý...")
        self.progress_label.setObjectName("progressLabel")
        self.progress_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("modernProgress")
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)
        return widget

    def create_results_area(self):
        widget = QWidget()
        widget.setObjectName("resultsArea")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        card = QWidget()
        card.setObjectName("resultCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(30, 30, 30, 30)
        card_layout.setSpacing(16)

        self.result_display = QTextEdit()
        self.result_display.setObjectName("resultDisplay")
        self.result_display.setReadOnly(True)
        self.result_display.setMinimumHeight(340)
        card_layout.addWidget(self.result_display)

        btn_row = QHBoxLayout()
        new_btn = QPushButton("Phân tích video khác")
        new_btn.setObjectName("secondaryButton")
        new_btn.clicked.connect(self.reset_ui)
        btn_row.addWidget(new_btn)
        btn_row.addStretch()
        card_layout.addLayout(btn_row)

        layout.addWidget(card)
        return widget

    # ── LOGIC ─────────────────────────────────

    def switch_mode(self, mode: str):
        self._input_mode = mode
        if mode == "file":
            self.stack.setCurrentIndex(0)
            self.btn_file_mode.setObjectName("modeActive")
            self.btn_url_mode.setObjectName("modeInactive")
        else:
            self.stack.setCurrentIndex(1)
            self.btn_file_mode.setObjectName("modeInactive")
            self.btn_url_mode.setObjectName("modeActive")
        for btn in [self.btn_file_mode, self.btn_url_mode]:
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def browse_file(self):
        if self._input_mode != "file":
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn video", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)"
        )
        if path:
            self._video_path = path
            self.file_selected_label.setText(f"✓ Đã chọn: {Path(path).name}")
            self.analyze_file_btn.setEnabled(True)

    def paste_url(self):
        text = QApplication.clipboard().text().strip()
        if text:
            self.url_input.setText(text)

    def on_url_changed(self, text: str):
        ok = text.strip().startswith("http://") or text.strip().startswith("https://")
        self.analyze_url_btn.setEnabled(ok)
        if ok:
            self.url_status_label.setText("✅ URL hợp lệ")
            self.url_status_label.setStyleSheet("color:#16a34a;font-size:13px;")
        else:
            self.url_status_label.setText("⚠️ Nhập URL bắt đầu bằng https://")
            self.url_status_label.setStyleSheet("color:#ca8a04;font-size:13px;")

    def start_file_analysis(self):
        if not self._video_path:
            return
        self._lock_ui()
        self._run_analysis(self._video_path)

    def start_url_analysis(self):
        url = self.url_input.text().strip()
        if not url:
            return
        self._lock_ui()
        self.progress_label.setText("Đang download video...")
        self._dl = DownloadThread(url, TEMP_DIR)
        self._dl.progress.connect(self.update_progress)
        self._dl.finished.connect(self._on_download_done)
        self._dl.error.connect(self.show_error)
        self._dl.start()

    def _on_download_done(self, path: str):
        self._video_path = path
        self.progress_label.setText("Download xong! Đang phân tích...")
        self._run_analysis(path)

    def _run_analysis(self, path: str):
        self._at = AnalysisThread(path)
        self._at.progress.connect(self.update_progress)
        self._at.finished.connect(self.show_results)
        self._at.error.connect(self.show_error)
        self._at.start()

    def _lock_ui(self):
        self.results_widget.setVisible(False)
        self.progress_widget.setVisible(True)
        self.analyze_file_btn.setEnabled(False)
        self.analyze_url_btn.setEnabled(False)

    def _unlock_ui(self):
        self.analyze_file_btn.setEnabled(self._video_path is not None)
        url = self.url_input.text().strip()
        self.analyze_url_btn.setEnabled(url.startswith("http"))

    def update_progress(self, msg: str):
        self.progress_label.setText(msg)

    def show_results(self, result: dict):
        self.progress_widget.setVisible(False)
        self.results_widget.setVisible(True)
        self._unlock_ui()

        self.history.append(result)
        self.save_history()

        prediction = result.get("prediction", "UNKNOWN")
        prob_fake  = result.get("probability_fake", 0) * 100
        prob_real  = result.get("probability_real", 0) * 100
        confidence = result.get("confidence", "N/A")
        color  = "#dc2626" if prediction == "FAKE" else "#16a34a"
        emoji  = "🚨" if prediction == "FAKE" else "✅"

        src = result.get("video_path", "")
        if src.startswith("http"):
            src_html = f'<div style="font-size:13px;color:#6e6e80;margin:8px 0;">🔗 Nguồn: {src[:90]}</div>'
        else:
            src_html = f'<div style="font-size:13px;color:#6e6e80;margin:8px 0;">📂 File: {Path(src).name}</div>'

        html = f"""
        <div style="text-align:center;padding:28px 20px;background:{color};color:white;border-radius:10px;margin-bottom:16px;">
            <div style="font-size:52px;margin-bottom:8px;">{emoji}</div>
            <div style="font-size:28px;font-weight:700;margin-bottom:6px;">{prediction}</div>
            <div style="font-size:15px;opacity:0.9;">Confidence: {confidence}</div>
        </div>
        {src_html}
        <div style="background:#f9fafb;padding:18px;border-radius:8px;margin:12px 0;">
            <div style="font-size:15px;font-weight:600;color:#202123;margin-bottom:12px;">📊 Xác suất</div>
            <div style="margin-bottom:10px;">
                <div style="color:#6e6e80;font-size:13px;margin-bottom:4px;">Fake: {prob_fake:.1f}%</div>
                <div style="background:#e5e7eb;height:8px;border-radius:4px;overflow:hidden;">
                    <div style="background:#dc2626;height:100%;width:{min(prob_fake,100):.1f}%;"></div>
                </div>
            </div>
            <div>
                <div style="color:#6e6e80;font-size:13px;margin-bottom:4px;">Real: {prob_real:.1f}%</div>
                <div style="background:#e5e7eb;height:8px;border-radius:4px;overflow:hidden;">
                    <div style="background:#16a34a;height:100%;width:{min(prob_real,100):.1f}%;"></div>
                </div>
            </div>
        </div>
        <div style="background:#f9fafb;padding:18px;border-radius:8px;margin:12px 0;">
            <div style="font-size:15px;font-weight:600;color:#202123;margin-bottom:10px;">🔍 Chi tiết</div>
            <div style="color:#565869;font-size:13px;line-height:1.8;">
                <div><strong>Artifact Score:</strong> {result.get('artifact_score',0):.3f}</div>
                <div><strong>Reality Score:</strong> {result.get('reality_score',0):.3f}</div>
            </div>
        </div>
        <div style="background:#f9fafb;padding:18px;border-radius:8px;">
            <div style="font-size:15px;font-weight:600;color:#202123;margin-bottom:10px;">💡 Phân tích</div>
            <div style="color:#565869;font-size:13px;line-height:1.8;">
        """
        for exp in result.get("explanations", []):
            html += f"<div style='margin-bottom:6px;'>• {exp}</div>"
        html += "</div></div>"

        self.result_display.setHtml(html)

    def show_error(self, error: str):
        self.progress_widget.setVisible(False)
        self._unlock_ui()
        QMessageBox.critical(self, "Lỗi", error)

    def reset_ui(self):
        self.results_widget.setVisible(False)
        self.file_selected_label.setText("")
        self.analyze_file_btn.setEnabled(False)
        self._video_path = None

    def show_history(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Lịch sử phân tích")
        dlg.setMinimumSize(660, 420)
        layout = QVBoxLayout(dlg)
        lst = QListWidget()
        for item in reversed(self.history[-30:]):
            ts   = item.get("timestamp", "")[:19]
            pred = item.get("prediction", "?")
            src  = item.get("video_path", "Unknown")
            name = src if src.startswith("http") else Path(src).name
            emoji = "🚨" if pred == "FAKE" else "✅"
            lst.addItem(f"{emoji}  {ts}  —  {name[:55]}  →  {pred}")
        layout.addWidget(lst)
        close_btn = QPushButton("Đóng")
        close_btn.clicked.connect(dlg.close)
        layout.addWidget(close_btn)
        dlg.exec()

    def show_about(self):
        QMessageBox.information(
            self, "Về ứng dụng",
            f"<h3>Deepfake Detector</h3>"
            f"<p>Version 1.1.0</p>"
            f"<p>Server: {RENDER_API_URL}</p>"
            f"<p>Hỗ trợ phân tích qua file và link video trực tuyến.</p>"
        )

    def load_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def save_history(self):
        try:
            self.history = self.history[-50:]
            with open(self.history_file, "w") as f:
                json.dump(self.history, f, indent=2)
        except Exception:
            pass

    def apply_modern_style(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #ffffff; }
            #topBar { background-color: #f7f7f8; border-bottom: 1px solid #e5e5e5; }
            #appTitle { font-size: 18px; font-weight: 600; color: #202123; }
            #topBarButton {
                background: transparent; border: none; color: #565869;
                padding: 8px 16px; border-radius: 6px; font-size: 14px;
            }
            #topBarButton:hover { background-color: #ececf1; }
            #welcome  { font-size: 30px; font-weight: 700; color: #202123; }
            #subtitle { font-size: 15px; color: #565869; }
            #modeActive {
                background: #2563eb; color: white; border: none;
                padding: 10px 28px; font-size: 14px; font-weight: 600;
                border-radius: 8px 0 0 8px;
            }
            #modeInactive {
                background: #f3f4f6; color: #6b7280; border: 1px solid #d1d5db;
                padding: 10px 28px; font-size: 14px;
                border-radius: 0 8px 8px 0;
            }
            #modeInactive:hover { background: #e5e7eb; }
            #uploadArea {
                background: #ffffff; border: 2px dashed #d1d5db;
                border-radius: 12px; min-height: 260px;
            }
            #uploadArea:hover { border-color: #10a37f; background: #f9fafb; }
            #urlPanel {
                background: #ffffff; border: 2px dashed #d1d5db;
                border-radius: 12px; min-height: 260px;
            }
            #urlInput {
                padding: 12px 14px; border: 1.5px solid #d1d5db;
                border-radius: 8px; font-size: 14px; color: #202123;
                background: #fafafa;
            }
            #urlInput:focus { border-color: #2563eb; background: #ffffff; }
            #pasteButton {
                background: #f3f4f6; color: #374151; border: 1px solid #d1d5db;
                padding: 10px 18px; border-radius: 8px; font-size: 13px; min-width: 70px;
            }
            #pasteButton:hover { background: #e5e7eb; }
            #platformHint { font-size: 12px; color: #9ca3af; }
            #uploadIcon   { font-size: 60px; }
            #uploadText   { font-size: 15px; font-weight: 600; color: #202123; }
            #uploadSubtext { font-size: 13px; color: #6e6e80; }
            #selectedFile  { font-size: 13px; color: #10a37f; font-weight: 500; }
            #primaryButton {
                background: #10a37f; color: white; border: none;
                padding: 12px 28px; border-radius: 8px;
                font-size: 14px; font-weight: 600; min-width: 140px;
            }
            #primaryButton:hover { background: #0d8f6f; }
            #analyzeButton {
                background: #2563eb; color: white; border: none;
                padding: 12px 28px; border-radius: 8px;
                font-size: 14px; font-weight: 600; min-width: 200px;
            }
            #analyzeButton:hover:enabled { background: #1d4ed8; }
            #analyzeButton:disabled { background: #9ca3af; }
            #secondaryButton {
                background: #f7f7f8; color: #202123; border: 1px solid #d1d5db;
                padding: 10px 20px; border-radius: 8px;
                font-size: 14px; font-weight: 500;
            }
            #secondaryButton:hover { background: #ececf1; }
            #progressArea {
                background: #f9fafb; border-radius: 12px; border: 1px solid #e5e5e5;
            }
            #progressLabel { font-size: 15px; color: #202123; font-weight: 500; }
            #modernProgress {
                border: none; background: #e5e7eb;
                border-radius: 4px; max-height: 8px;
            }
            #modernProgress::chunk { background: #10a37f; border-radius: 4px; }
            #resultCard {
                background: #ffffff; border: 1px solid #e5e5e5; border-radius: 12px;
            }
            #resultDisplay {
                background: #ffffff; border: none; font-size: 14px; color: #202123;
            }
            QScrollArea { border: none; }
        """)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("Deepfake Detector")
    app.setOrganizationName("AI Research")
    window = ModernWindow()
    window.show()
    sys.exit(app.exec())