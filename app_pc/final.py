import json
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

if getattr(sys, "frozen", False):
    BASE_PATH = os.path.dirname(sys.executable)
else:
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))

os.chdir(BASE_PATH)
if BASE_PATH not in sys.path:
    sys.path.insert(0, BASE_PATH)

from src.aligned_inference import run_aligned_inference

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def discover_models(models_dir: Path):
    available = []
    for onnx_path in models_dir.glob("*.onnx"):
        stem = onnx_path.stem
        scaler_path = models_dir / f"{stem}_scaler.pkl"
        if scaler_path.exists():
            available.append(stem)
    return sorted(set(available))


def collect_videos(folder: Path):
    if not folder.exists():
        return []
    return sorted([str(path) for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTS])


def linspace_int(start: int, stop: int, count: int):
    if count <= 1:
        return [start]
    step = (stop - start) / float(count - 1)
    return [int(start + step * i) for i in range(count)]


def extract_preview_frames(video_path: str, count: int = 6):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        cap.release()
        return []

    frames = []
    for idx in linspace_int(0, max(total_frames - 1, 0), count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        frames.append((idx, image))
    cap.release()
    return frames


def suggest_scan_mode(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "accurate", "Accurate scan duoc giu lam mac dinh."
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration = frames / fps if fps > 0 else 0.0
    if duration and duration < 4.0:
        return "accurate", "Video ngan, nen uu tien Accurate scan de giu on dinh."
    if width * height >= 1920 * 1080 or duration > 90:
        return "quick", "Video dai hoac do phan giai cao, Quick scan se de chiu hon."
    return "accurate", "Accurate scan duoc goi y cho video muc tieu."


def pct_bar(value: float, color: str) -> str:
    safe = max(0.0, min(1.0, float(value)))
    width = max(4, int(safe * 100))
    return (
        "<div style='margin-top:8px;background:#0a1322;border:1px solid #223654;border-radius:999px;height:10px;overflow:hidden;'>"
        f"<div style='height:100%;width:{width}%;background:{color};border-radius:999px;'></div>"
        "</div>"
    )


class AnalysisThread(QThread):
    finished = Signal(dict)
    progress = Signal(str)
    error = Signal(str)

    def __init__(self, video_path: str, model_stem: str, config_path: str, scan_mode: str):
        super().__init__()
        self.video_path = video_path
        self.model_stem = model_stem
        self.config_path = config_path
        self.scan_mode = scan_mode

    def run(self):
        try:
            self.progress.emit(f"Running {self.scan_mode} offline analysis...")
            result = run_aligned_inference(self.video_path, self.model_stem, self.config_path, self.scan_mode)
            result["timestamp"] = datetime.now().isoformat()
            self.finished.emit(result)
        except Exception as exc:
            import traceback

            self.error.emit(f"{exc}\n\n{traceback.format_exc()}")


class BatchAnalysisThread(QThread):
    finished = Signal(list)
    progress = Signal(str)
    error = Signal(str)

    def __init__(self, video_paths: list[str], model_stem: str, config_path: str, scan_mode: str):
        super().__init__()
        self.video_paths = video_paths
        self.model_stem = model_stem
        self.config_path = config_path
        self.scan_mode = scan_mode

    def run(self):
        results = []
        try:
            total = len(self.video_paths)
            for index, video_path in enumerate(self.video_paths, start=1):
                self.progress.emit(f"[{index}/{total}] {Path(video_path).name}")
                try:
                    result = run_aligned_inference(video_path, self.model_stem, self.config_path, self.scan_mode)
                    result["timestamp"] = datetime.now().isoformat()
                    results.append(result)
                except Exception as exc:
                    results.append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "video_path": video_path,
                            "model_stem": self.model_stem,
                            "scan_mode": self.scan_mode,
                            "customer_verdict": "ERROR",
                            "prediction": "ERROR",
                            "confidence": "LOW",
                            "reason_summary": str(exc),
                            "probability_fake": 0.0,
                        }
                    )
            self.finished.emit(results)
        except Exception as exc:
            import traceback

            self.error.emit(f"{exc}\n\n{traceback.format_exc()}")


class BenchmarkThread(QThread):
    finished = Signal(dict)
    progress = Signal(str)
    error = Signal(str)

    def __init__(self, models_dir: str, data_dir: str, config_path: str, limit_per_class: int):
        super().__init__()
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.config_path = config_path
        self.limit_per_class = limit_per_class

    def run(self):
        try:
            dataset = []
            for label_name, y in [("real", 0), ("fake", 1)]:
                class_dir = self.data_dir / label_name
                videos = sorted([p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS]) if class_dir.exists() else []
                if self.limit_per_class > 0:
                    videos = videos[: self.limit_per_class]
                dataset.extend((str(video), y) for video in videos)

            if not dataset:
                raise ValueError("No dataset files found in data/real and data/fake")

            models = discover_models(self.models_dir)
            if not models:
                raise ValueError("No ONNX + scaler pairs found")

            results = {}
            best_model = None
            best_key = (-1.0, -1.0, -1.0)
            for model in models:
                self.progress.emit(f"Benchmarking {model} on {len(dataset)} videos...")
                y_true = []
                y_pred = []
                for video_path, label in dataset:
                    output = run_aligned_inference(video_path, model, self.config_path, "accurate")
                    y_true.append(label)
                    y_pred.append(1 if output["prediction"] == "FAKE" else 0)

                total = len(y_true)
                correct = sum(1 for truth, pred in zip(y_true, y_pred) if truth == pred)
                tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 1)
                fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 1)
                fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 0)
                precision = tp / (tp + fp) if (tp + fp) else 0.0
                recall = tp / (tp + fn) if (tp + fn) else 0.0
                f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
                accuracy = correct / total if total else 0.0
                results[model] = {"samples": total, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
                ranking = (f1, accuracy, recall)
                if ranking > best_key:
                    best_key = ranking
                    best_model = model

            self.finished.emit(
                {
                    "best_model": best_model,
                    "reason": "selected_by_desktop_dashboard",
                    "ranking_metric": ["f1", "accuracy", "recall"],
                    "results": results,
                    "dataset_size": len(dataset),
                    "generated_at": datetime.now().isoformat(),
                }
            )
        except Exception as exc:
            import traceback

            self.error.emit(f"{exc}\n\n{traceback.format_exc()}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("AI Checker Offline")
        self.setMinimumSize(1320, 860)

        self.config_path = os.path.join(BASE_PATH, "config.yaml")
        self.models_dir = os.path.join(BASE_PATH, "models")
        self.training_summary_path = os.path.join(self.models_dir, "training_summary.json")
        self.benchmark_best_path = os.path.join(self.models_dir, "benchmark_best_model.json")
        self.history_file = os.path.join(BASE_PATH, "output", "history.json")
        self.batch_output_file = os.path.join(BASE_PATH, "output", "batch_last_run.json")
        self.selected_path = None
        self.selected_batch_folder = None
        self.last_result = None
        self.last_batch_results = []
        self.history = self.load_history()
        self.recommended_model = self._read_recommended_model()
        self.user_profile = "accuracy"

        self.setup_theme()
        self.setup_ui()
        self.load_models()
        self.refresh_history()

    def setup_theme(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #09111f; color: #e5eef8; font-family: 'Segoe UI'; }
            QGroupBox {
                background: #0f1a2d;
                border: 1px solid #21314d; border-radius: 18px; margin-top: 14px; padding-top: 18px;
                color: #b3d4ff; font-weight: 700;
            }
            QGroupBox::title { left: 16px; padding: 0 8px; }
            QLabel#title { font-size: 34px; font-weight: 900; color: #f8fbff; }
            QLabel#subtitle { font-size: 14px; color: #8ea3bf; }
            QLabel#banner {
                background: #10233e; color: #d8e7fb; border: 1px solid #29466f;
                border-radius: 16px; padding: 14px 18px; font-size: 13px;
            }
            QLabel#thumb {
                background: #122038; border: 1px solid #2b3f5e; border-radius: 16px; padding: 8px;
            }
            QComboBox, QTextEdit, QTableWidget, QLineEdit, QSpinBox {
                background: #122038; border: 1px solid #2a3c58; border-radius: 14px; color: #f8fbff; padding: 10px;
            }
            QPushButton {
                background: #1f6feb; color: white; border: none; border-radius: 14px;
                padding: 12px 18px; font-weight: 800;
            }
            QPushButton:hover { background: #2d7ef7; }
            QPushButton:disabled { background: #3b4b62; color: #c5d2e2; }
            QPushButton#secondary {
                background: #122038; border: 1px solid #31455f; color: #e2ebf5;
            }
            QPushButton#secondary:hover { background: #1a2d47; }
            QProgressBar {
                background: #122038; border: 1px solid #2a3c58; border-radius: 10px; min-height: 18px; color: white;
            }
            QProgressBar::chunk { background: #19c37d; border-radius: 9px; }
            QHeaderView::section {
                background: #112038; color: #d5e4f5; border: none; border-bottom: 1px solid #2a3c58; padding: 10px; font-weight: 800;
            }
            QTableWidget::item:selected { background: #1f5fd4; }
            QTabWidget::pane {
                border: 1px solid #1d2d47;
                border-radius: 18px;
                top: -1px;
                background: #0c1525;
            }
            QTabBar::tab {
                background: #0f1a2d;
                color: #8ea3bf;
                padding: 12px 18px;
                margin-right: 6px;
                border-top-left-radius: 12px;
                border-top-right-radius: 12px;
            }
            QTabBar::tab:selected {
                background: #153054;
                color: #f7fbff;
            }
            """
        )

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(18)

        brand_row = QHBoxLayout()
        brand_mark = QLabel("AC")
        brand_mark.setStyleSheet(
            "background:#1f6feb;color:white;font-weight:900;border-radius:16px;padding:10px 14px;font-size:18px;"
        )
        brand_mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brand_col = QVBoxLayout()
        title = QLabel("AIChecker Pro")
        title.setObjectName("title")
        subtitle = QLabel("Offline forensic screening cho video: nhanh hon de van hanh, ro hon de ra quyet dinh, dep hon de trien khai nhu mot san pham thuong mai.")
        subtitle.setObjectName("subtitle")
        brand_col.addWidget(title)
        brand_col.addWidget(subtitle)
        brand_row.addWidget(brand_mark)
        brand_row.addLayout(brand_col, 1)
        self.banner = QLabel("Best setup auto-enabled: app uu tien model benchmark tot nhat, profile accuracy-first va recommendation thong minh theo video.")
        self.banner.setObjectName("banner")
        root.addLayout(brand_row)
        root.addWidget(self.banner)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        root.addWidget(self.tabs)

        self.setup_analyze_tab()
        self.setup_batch_tab()
        self.setup_history_tab()
        self.setup_benchmark_tab()
        self.setup_about_tab()
        self.statusBar().showMessage("Ready")

    def setup_analyze_tab(self):
        tab = QWidget()
        outer = QHBoxLayout(tab)
        outer.setSpacing(18)
        left = QVBoxLayout()
        right = QVBoxLayout()
        left.setSpacing(16)
        right.setSpacing(16)
        outer.addLayout(left, 3)
        outer.addLayout(right, 2)

        input_group = QGroupBox("Video")
        input_layout = QVBoxLayout(input_group)
        file_row = QHBoxLayout()
        self.file_label = QLabel("No local file selected.")
        self.file_label.setObjectName("subtitle")
        self.browse_btn = QPushButton("Choose File")
        self.browse_btn.setObjectName("secondary")
        self.browse_btn.clicked.connect(self.browse_file)
        file_row.addWidget(self.file_label, 1)
        file_row.addWidget(self.browse_btn)
        input_layout.addLayout(file_row)
        self.scan_hint = QLabel("Scan recommendation will appear here after choosing a file.")
        self.scan_hint.setObjectName("subtitle")
        input_layout.addWidget(self.scan_hint)
        left.addWidget(input_group)

        model_group = QGroupBox("Model")
        model_layout = QGridLayout(model_group)
        model_layout.addWidget(QLabel("Active model"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        model_layout.addWidget(self.model_combo, 0, 1)
        self.model_detail = QLabel("")
        self.model_detail.setObjectName("subtitle")
        model_layout.addWidget(self.model_detail, 1, 0, 1, 2)
        self.model_note = QLabel("")
        self.model_note.setObjectName("subtitle")
        model_layout.addWidget(self.model_note, 2, 0, 1, 2)
        model_layout.addWidget(QLabel("Usage profile"), 3, 0)
        self.profile_combo = QComboBox()
        self.profile_combo.addItem("Can bang", "balanced")
        self.profile_combo.addItem("Uu tien do chinh xac", "accuracy")
        self.profile_combo.addItem("Uu tien toc do", "speed")
        self.profile_combo.currentIndexChanged.connect(self.on_profile_changed)
        model_layout.addWidget(self.profile_combo, 3, 1)
        self.profile_combo.setCurrentIndex(1)
        self.scan_combo = QComboBox()
        self.scan_combo.addItem("Quick scan", "quick")
        self.scan_combo.addItem("Accurate scan", "accurate")
        self.scan_combo.setCurrentIndex(1)
        model_layout.addWidget(self.scan_combo, 4, 1)
        model_layout.addWidget(QLabel("Scan mode"), 4, 0)
        self.scan_note = QLabel("Quick scan nhanh hon. Accurate scan lay nhieu frame hon de on dinh ket qua.")
        self.scan_note.setObjectName("subtitle")
        model_layout.addWidget(self.scan_note, 5, 0, 1, 2)
        self.recommendation_note = QLabel("App se goi y model va scan mode theo muc tieu su dung.")
        self.recommendation_note.setObjectName("subtitle")
        model_layout.addWidget(self.recommendation_note, 6, 0, 1, 2)
        left.addWidget(model_group)

        action_group = QGroupBox("Actions")
        action_layout = QHBoxLayout(action_group)
        self.analyze_btn = QPushButton("Run Offline Analysis")
        self.analyze_btn.clicked.connect(self.start_analysis)
        self.analyze_btn.setEnabled(False)
        self.export_report_btn = QPushButton("Export HTML Report")
        self.export_report_btn.setObjectName("secondary")
        self.export_report_btn.clicked.connect(self.export_single_report)
        self.export_report_btn.setEnabled(False)
        action_layout.addWidget(self.analyze_btn)
        action_layout.addWidget(self.export_report_btn)
        left.addWidget(action_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        self.progress_label = QLabel("")
        self.progress_label.setObjectName("subtitle")
        left.addWidget(self.progress_bar)
        left.addWidget(self.progress_label)

        result_group = QGroupBox("Result")
        result_layout = QVBoxLayout(result_group)
        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        self.result_display.setMinimumHeight(420)
        result_layout.addWidget(self.result_display)
        left.addWidget(result_group)

        preview_group = QGroupBox("Preview Frames")
        preview_layout = QVBoxLayout(preview_group)
        preview_label = QLabel("Frame strip se cap nhat ngay sau khi chon video.")
        preview_label.setObjectName("subtitle")
        preview_layout.addWidget(preview_label)
        preview_scroll = QScrollArea()
        preview_scroll.setWidgetResizable(True)
        preview_container = QWidget()
        self.preview_row = QHBoxLayout(preview_container)
        self.preview_row.setSpacing(10)
        self.preview_row.setContentsMargins(0, 0, 0, 0)
        preview_scroll.setWidget(preview_container)
        preview_layout.addWidget(preview_scroll)
        right.addWidget(preview_group)

        self.tabs.addTab(tab, "Analyze")

    def setup_batch_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(16)

        folder_group = QGroupBox("Batch Folder")
        folder_layout = QVBoxLayout(folder_group)
        folder_row = QHBoxLayout()
        self.batch_folder_label = QLabel("No folder selected.")
        self.batch_folder_label.setObjectName("subtitle")
        self.batch_choose_btn = QPushButton("Choose Folder")
        self.batch_choose_btn.setObjectName("secondary")
        self.batch_choose_btn.clicked.connect(self.choose_batch_folder)
        self.batch_run_btn = QPushButton("Run Batch Analysis")
        self.batch_run_btn.clicked.connect(self.start_batch_analysis)
        self.batch_run_btn.setEnabled(False)
        self.batch_export_btn = QPushButton("Export Batch Report")
        self.batch_export_btn.setObjectName("secondary")
        self.batch_export_btn.clicked.connect(self.export_batch_report)
        self.batch_export_btn.setEnabled(False)
        folder_row.addWidget(self.batch_folder_label, 1)
        folder_row.addWidget(self.batch_choose_btn)
        folder_row.addWidget(self.batch_run_btn)
        folder_row.addWidget(self.batch_export_btn)
        folder_layout.addLayout(folder_row)
        self.batch_summary = QLabel("Batch mode phu hop khi can quet ca thu muc video.")
        self.batch_summary.setObjectName("subtitle")
        folder_layout.addWidget(self.batch_summary)
        layout.addWidget(folder_group)

        self.batch_progress = QProgressBar()
        self.batch_progress.hide()
        self.batch_progress_label = QLabel("")
        self.batch_progress_label.setObjectName("subtitle")
        layout.addWidget(self.batch_progress)
        layout.addWidget(self.batch_progress_label)

        self.batch_table = QTableWidget()
        self.batch_table.setColumnCount(6)
        self.batch_table.setHorizontalHeaderLabels(["Video", "Verdict", "Confidence", "Fake %", "Model", "Summary"])
        self.batch_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.batch_table)
        self.batch_empty = QLabel("Batch queue chua co du lieu. Chon mot thu muc de quet nhieu video trong mot lan.")
        self.batch_empty.setObjectName("subtitle")
        layout.addWidget(self.batch_empty)
        self.tabs.addTab(tab, "Batch")

    def setup_history_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(16)

        filter_group = QGroupBox("History Filters")
        filter_layout = QHBoxLayout(filter_group)
        self.history_search = QLineEdit()
        self.history_search.setPlaceholderText("Search by file name or model...")
        self.history_search.textChanged.connect(self.refresh_history)
        self.history_filter = QComboBox()
        self.history_filter.addItems(["All verdicts", "LIKELY_FAKE", "LIKELY_REAL", "UNCERTAIN", "INSUFFICIENT_QUALITY", "ERROR"])
        self.history_filter.currentTextChanged.connect(self.refresh_history)
        self.history_export_btn = QPushButton("Export History JSON")
        self.history_export_btn.setObjectName("secondary")
        self.history_export_btn.clicked.connect(self.export_history_json)
        filter_layout.addWidget(self.history_search, 2)
        filter_layout.addWidget(self.history_filter)
        filter_layout.addWidget(self.history_export_btn)
        layout.addWidget(filter_group)

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels(["Time", "Video", "Model / Mode", "Verdict", "Confidence"])
        self.history_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.history_table)
        self.history_empty = QLabel("History dang trong. Sau lan phan tich dau tien, ket qua se hien o day.")
        self.history_empty.setObjectName("subtitle")
        layout.addWidget(self.history_empty)
        self.tabs.addTab(tab, "History")

    def setup_benchmark_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(16)

        control_group = QGroupBox("Benchmark")
        control_layout = QGridLayout(control_group)
        control_layout.addWidget(QLabel("Dataset folder"), 0, 0)
        self.benchmark_dir_edit = QLineEdit(str(Path(BASE_PATH) / "data"))
        control_layout.addWidget(self.benchmark_dir_edit, 0, 1)
        self.benchmark_dir_btn = QPushButton("Choose")
        self.benchmark_dir_btn.setObjectName("secondary")
        self.benchmark_dir_btn.clicked.connect(self.choose_benchmark_folder)
        control_layout.addWidget(self.benchmark_dir_btn, 0, 2)
        control_layout.addWidget(QLabel("Limit per class"), 1, 0)
        self.benchmark_limit = QSpinBox()
        self.benchmark_limit.setRange(0, 50000)
        self.benchmark_limit.setValue(10)
        self.benchmark_limit.setSpecialValueText("All")
        control_layout.addWidget(self.benchmark_limit, 1, 1)
        self.benchmark_btn = QPushButton("Run Benchmark")
        self.benchmark_btn.clicked.connect(self.run_benchmark)
        control_layout.addWidget(self.benchmark_btn, 1, 2)
        layout.addWidget(control_group)

        self.benchmark_progress = QProgressBar()
        self.benchmark_progress.hide()
        self.benchmark_status = QLabel("Benchmark dashboard cho phep chon best model ngay trong app.")
        self.benchmark_status.setObjectName("subtitle")
        self.benchmark_output = QTextEdit()
        self.benchmark_output.setReadOnly(True)
        layout.addWidget(self.benchmark_progress)
        layout.addWidget(self.benchmark_status)
        layout.addWidget(self.benchmark_output)
        self.benchmark_empty = QLabel("Chua co benchmark moi. Ban co the chay benchmark de cap nhat model khuyen nghi.")
        self.benchmark_empty.setObjectName("subtitle")
        layout.addWidget(self.benchmark_empty)
        self.tabs.addTab(tab, "Benchmark")

    def setup_about_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        about = QTextEdit()
        about.setReadOnly(True)
        about.setHtml(
            f"""
            <h2>Offline Desktop Build</h2>
            <p>This app no longer depends on any server/client/cloud files.</p>
            <p><b>Config:</b> {self.config_path}</p>
            <p><b>Models dir:</b> {self.models_dir}</p>
            <p><b>Recommended model:</b> {self.recommended_model or 'Not found'}</p>
            <p><b>What is new:</b> single scan, batch scan, history filters, report export, preview frame strip, benchmark dashboard.</p>
            """
        )
        layout.addWidget(about)
        self.tabs.addTab(tab, "About")

    def _read_recommended_model(self):
        if os.path.exists(self.benchmark_best_path):
            try:
                with open(self.benchmark_best_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                best_model = data.get("best_model")
                if best_model:
                    return best_model
            except Exception:
                pass

        if not os.path.exists(self.training_summary_path):
            return None
        try:
            with open(self.training_summary_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            model_path = data.get("model_path", "")
            return Path(model_path).stem if model_path else None
        except Exception:
            return None

    def load_models(self):
        available = discover_models(Path(self.models_dir))
        self.model_combo.clear()
        self.model_combo.addItems(available)
        if self.recommended_model and self.recommended_model in available:
            self.model_combo.setCurrentText(self.recommended_model)
        elif "onestar" in available:
            self.model_combo.setCurrentText("onestar")
        elif available:
            self.model_combo.setCurrentIndex(0)
        self.on_model_changed(self.model_combo.currentText())

    def on_model_changed(self, model_stem: str):
        if not model_stem:
            self.model_detail.setText("No offline model available.")
            self.model_note.setText("")
            self.analyze_btn.setEnabled(False)
            self.batch_run_btn.setEnabled(False)
            return

        scaler_path = Path(self.models_dir) / f"{model_stem}_scaler.pkl"
        deep_resnet = Path(self.models_dir) / "resnet50_features.onnx"
        deep_eff = Path(self.models_dir) / "efficientnet_b0_features.onnx"
        self.model_detail.setText(f"Using {model_stem}.onnx + {scaler_path.name}")
        if model_stem == self.recommended_model:
            self.model_note.setText("Benchmark winner. Day la model duoc uu tien cho ban thuong mai mac dinh.")
        elif deep_resnet.exists() or deep_eff.exists():
            self.model_note.setText("Deep feature assets da san sang. Do on dinh va parity voi Android se tot hon.")
        else:
            self.model_note.setText("Thieu deep assets. Nen uu tien model benchmark winner neu can do tin cay cao.")

        self.analyze_btn.setEnabled(self.selected_path is not None)
        self.batch_run_btn.setEnabled(self.selected_batch_folder is not None)
        self.update_recommendation_context()

    def on_profile_changed(self):
        self.user_profile = self.profile_combo.currentData() or "balanced"
        self.update_recommendation_context()

    def update_recommendation_context(self):
        model = self.model_combo.currentText()
        if not model:
            return

        suggested_mode, scan_reason = ("accurate", "Accurate scan duoc giu lam mac dinh.")
        if self.selected_path:
            suggested_mode, scan_reason = suggest_scan_mode(self.selected_path)

        recommended_model = model
        recommended_mode = suggested_mode
        profile = self.user_profile

        if profile == "accuracy":
            recommended_model = self.recommended_model or model
            recommended_mode = "accurate"
            profile_text = "Ho so nay uu tien do chinh xac va do on dinh."
        elif profile == "speed":
            recommended_mode = "quick"
            profile_text = "Ho so nay uu tien tra ket qua nhanh va quet nhieu video."
        else:
            recommended_model = self.recommended_model or model
            recommended_mode = suggested_mode
            profile_text = "Ho so nay can bang giua toc do va do chinh xac."

        if recommended_model and recommended_model != model and self.model_combo.findText(recommended_model) >= 0:
            self.model_combo.blockSignals(True)
            self.model_combo.setCurrentText(recommended_model)
            self.model_combo.blockSignals(False)
            model = recommended_model
            self.model_detail.setText(f"Using {model}.onnx + {model}_scaler.pkl")

        current_mode = self.scan_combo.currentData()
        if recommended_mode != current_mode:
            self.scan_combo.blockSignals(True)
            self.scan_combo.setCurrentIndex(0 if recommended_mode == "quick" else 1)
            self.scan_combo.blockSignals(False)

        mode_label = "Quick scan" if recommended_mode == "quick" else "Accurate scan"
        self.recommendation_note.setText(
            f"Commercial preset: {model} + {mode_label}. {profile_text} {scan_reason}"
        )

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose local video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)",
        )
        if not file_path:
            return

        self.selected_path = file_path
        self.file_label.setText(Path(file_path).name)
        self.analyze_btn.setEnabled(bool(self.model_combo.currentText()))
        mode, note = suggest_scan_mode(file_path)
        self.scan_combo.setCurrentIndex(0 if mode == "quick" else 1)
        self.scan_hint.setText(note)
        self.render_preview_strip(file_path)
        self.update_recommendation_context()

    def render_preview_strip(self, video_path: str):
        while self.preview_row.count():
            item = self.preview_row.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        frames = extract_preview_frames(video_path, count=6)
        if not frames:
            label = QLabel("Preview unavailable for this file.")
            label.setObjectName("subtitle")
            self.preview_row.addWidget(label)
            return

        for frame_idx, image in frames:
            pixmap = QPixmap.fromImage(image).scaled(180, 110, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            wrapper = QWidget()
            wrapper_layout = QVBoxLayout(wrapper)
            wrapper_layout.setContentsMargins(0, 0, 0, 0)
            thumb = QLabel()
            thumb.setObjectName("thumb")
            thumb.setPixmap(pixmap)
            thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
            caption = QLabel(f"Frame {frame_idx}")
            caption.setObjectName("subtitle")
            caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
            wrapper_layout.addWidget(thumb)
            wrapper_layout.addWidget(caption)
            self.preview_row.addWidget(wrapper)
        self.preview_row.addStretch(1)

    def start_analysis(self):
        if not self.selected_path:
            QMessageBox.warning(self, "Missing file", "Choose a local video first.")
            return

        self.analyze_btn.setEnabled(False)
        self.progress_bar.show()
        self.progress_bar.setRange(0, 0)
        self.progress_label.setText("Preparing offline analysis...")
        self.result_display.clear()

        self.analysis_thread = AnalysisThread(
            self.selected_path,
            self.model_combo.currentText(),
            self.config_path,
            self.scan_combo.currentData(),
        )
        self.analysis_thread.progress.connect(self.update_progress)
        self.analysis_thread.finished.connect(self.on_analysis_finished)
        self.analysis_thread.error.connect(self.on_error)
        self.analysis_thread.start()

    def update_progress(self, message: str):
        self.progress_label.setText(message)
        self.statusBar().showMessage(message)

    def on_analysis_finished(self, result: dict):
        self.progress_bar.hide()
        self.progress_label.clear()
        self.analyze_btn.setEnabled(True)
        self.export_report_btn.setEnabled(True)
        self.last_result = result
        self.display_result(result)
        self.save_to_history(result)
        self.refresh_history()
        self.statusBar().showMessage(f"Completed: {result.get('customer_verdict', result['prediction'])}")

    def on_error(self, error_msg: str):
        self.progress_bar.hide()
        self.progress_label.clear()
        self.analyze_btn.setEnabled(True)
        QMessageBox.critical(self, "Analysis error", error_msg)
        self.statusBar().showMessage("Error")

    def display_result(self, result: dict):
        prediction = result["prediction"]
        prob_fake = result["probability_fake"]
        prob_real = result["probability_real"]
        model_prob = result.get("model_probability_fake", prob_fake)
        fusion_prob = result.get("fusion_result", {}).get("final_probability", prob_fake)
        input_quality = result.get("input_quality_label", "MEDIUM")
        input_quality_score = result.get("input_quality_score", 0.75)
        confidence = result.get("confidence", result.get("fusion_result", {}).get("confidence", "N/A"))
        customer_verdict = result.get("customer_verdict", prediction)
        headline = result.get("verdict_headline", prediction)
        reason_summary = result.get("reason_summary", headline)
        reason_points = result.get("reason_points", [])
        scan_profile = result.get("scan_profile", {})

        if customer_verdict == "LIKELY_FAKE":
            color = "#ef4444"
            badge_text = "AI risk high"
        elif customer_verdict == "LIKELY_REAL":
            color = "#22c55e"
            badge_text = "Looks authentic"
        elif customer_verdict == "INSUFFICIENT_QUALITY":
            color = "#f59e0b"
            badge_text = "Quality too low"
        else:
            color = "#f97316"
            badge_text = "Needs review"

        html = f"""
        <div style="font-family:Segoe UI;color:#e6eef9;">
            <div style="background:{color};padding:28px;border-radius:18px;margin-bottom:16px;">
                <div style="font-size:12px;font-weight:800;letter-spacing:1.4px;opacity:.95;text-transform:uppercase;">{badge_text}</div>
                <div style="font-size:34px;font-weight:900;margin-top:6px;">{headline}</div>
                <div style="font-size:15px;opacity:.96;margin-top:8px;">{reason_summary}</div>
                <div style="font-size:13px;opacity:.88;margin-top:12px;">Confidence: {confidence} | Model: {result.get('model_stem', 'N/A')} | Mode: {scan_profile.get('label', 'Accurate scan')}</div>
            </div>
            <table width="100%" cellspacing="0" cellpadding="0" style="margin-bottom:12px;">
                <tr>
                    <td style="width:33%;vertical-align:top;padding-right:8px;">
                        <div style="background:#122038;padding:16px;border-radius:16px;border:1px solid #29415e;">
                            <div style="font-size:11px;color:#8fb3da;text-transform:uppercase;font-weight:800;">Final Verdict</div>
                            <div style="font-size:26px;font-weight:900;margin-top:6px;">{customer_verdict}</div>
                            <div style="font-size:13px;color:#a8bdd7;margin-top:6px;">Customer-facing decision layer</div>
                        </div>
                    </td>
                    <td style="width:33%;vertical-align:top;padding:0 4px;">
                        <div style="background:#122038;padding:16px;border-radius:16px;border:1px solid #29415e;">
                            <div style="font-size:11px;color:#8fb3da;text-transform:uppercase;font-weight:800;">Fake Probability</div>
                            <div style="font-size:26px;font-weight:900;margin-top:6px;">{prob_fake:.1%}</div>
                            <div style="font-size:13px;color:#a8bdd7;margin-top:6px;">Blended from model + fusion</div>
                            {pct_bar(prob_fake, '#ef4444')}
                        </div>
                    </td>
                    <td style="width:33%;vertical-align:top;padding-left:8px;">
                        <div style="background:#122038;padding:16px;border-radius:16px;border:1px solid #29415e;">
                            <div style="font-size:11px;color:#8fb3da;text-transform:uppercase;font-weight:800;">Input Quality</div>
                            <div style="font-size:26px;font-weight:900;margin-top:6px;">{input_quality}</div>
                            <div style="font-size:13px;color:#a8bdd7;margin-top:6px;">Score {input_quality_score:.2f}</div>
                            {pct_bar(input_quality_score, '#19c37d')}
                        </div>
                    </td>
                </tr>
            </table>
            <div style="background:#122038;padding:18px;border-radius:16px;border:1px solid #29415e;margin-bottom:12px;">
                <div style="font-size:12px;color:#8fb3da;text-transform:uppercase;font-weight:800;margin-bottom:10px;">Scoreboard</div>
                <div style="margin-bottom:6px;">Final blended fake score: <b>{prob_fake:.1%}</b></div>
                {pct_bar(prob_fake, '#ef4444')}
                <div style="margin-bottom:6px;">ONNX model fake score: <b>{model_prob:.1%}</b></div>
                {pct_bar(model_prob, '#f97316')}
                <div style="margin-bottom:6px;">Rule fusion fake score: <b>{fusion_prob:.1%}</b></div>
                {pct_bar(fusion_prob, '#60a5fa')}
                <div>Real score: <b>{prob_real:.1%}</b></div>
                {pct_bar(prob_real, '#19c37d')}
            </div>
            <div style="background:#122038;padding:18px;border-radius:16px;border:1px solid #29415e;margin-bottom:12px;">
                <div style="font-size:12px;color:#8fb3da;text-transform:uppercase;font-weight:800;margin-bottom:10px;">Diagnostics</div>
                <div style="margin-bottom:6px;">Artifact score: <b>{result.get('artifact_score', 0):.3f}</b></div>
                <div style="margin-bottom:6px;">Reality score: <b>{result.get('reality_score', 0):.3f}</b></div>
                <div style="margin-bottom:6px;">Stress proxy: <b>{result.get('stress_score', 0):.3f}</b></div>
                <div>Frames: <b>{result.get('metadata', {}).get('num_frames', 'N/A')}</b> | Resolution: <b>{result.get('metadata', {}).get('width', 'N/A')}x{result.get('metadata', {}).get('height', 'N/A')}</b> | Duration: <b>{result.get('metadata', {}).get('duration', 'N/A')}</b></div>
            </div>
            <div style="background:#122038;padding:18px;border-radius:16px;border:1px solid #29415e;">
                <div style="font-size:12px;color:#8fb3da;text-transform:uppercase;font-weight:800;margin-bottom:10px;">Findings</div>
        """

        for point in reason_points:
            html += f"<div style='margin-bottom:6px;'>- {point}</div>"
        for warning in result.get("quality_flags", []):
            html += f"<div style='margin-bottom:6px;color:#fbbf24;'>Input warning: {warning}</div>"
        for exp in result.get("explanations", []):
            html += f"<div style='margin-bottom:6px;'>- {exp}</div>"
        html += "</div></div>"
        self.result_display.setHtml(html)

    def choose_batch_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose folder with videos", "")
        if not folder:
            return
        self.selected_batch_folder = folder
        videos = collect_videos(Path(folder))
        self.batch_folder_label.setText(folder)
        self.batch_summary.setText(f"Found {len(videos)} video files ready for batch analysis.")
        self.batch_run_btn.setEnabled(bool(videos) and bool(self.model_combo.currentText()))

    def start_batch_analysis(self):
        if not self.selected_batch_folder:
            QMessageBox.warning(self, "Missing folder", "Choose a folder first.")
            return
        videos = collect_videos(Path(self.selected_batch_folder))
        if not videos:
            QMessageBox.warning(self, "No videos", "No supported video files found in that folder.")
            return

        self.batch_progress.show()
        self.batch_progress.setRange(0, 0)
        self.batch_progress_label.setText(f"Preparing batch run for {len(videos)} videos...")
        self.batch_run_btn.setEnabled(False)

        self.batch_thread = BatchAnalysisThread(
            videos,
            self.model_combo.currentText(),
            self.config_path,
            self.scan_combo.currentData(),
        )
        self.batch_thread.progress.connect(self.batch_progress_label.setText)
        self.batch_thread.finished.connect(self.on_batch_finished)
        self.batch_thread.error.connect(self.on_batch_error)
        self.batch_thread.start()

    def on_batch_finished(self, results: list[dict]):
        self.batch_progress.hide()
        self.batch_progress_label.setText(f"Batch complete: {len(results)} videos processed.")
        self.batch_run_btn.setEnabled(True)
        self.batch_export_btn.setEnabled(bool(results))
        self.batch_empty.setVisible(not bool(results))
        self.last_batch_results = results
        self.render_batch_results(results)
        if results:
            os.makedirs(os.path.dirname(self.batch_output_file), exist_ok=True)
            with open(self.batch_output_file, "w", encoding="utf-8") as file:
                json.dump(results, file, indent=2, ensure_ascii=False)
            for item in results[:25]:
                self.save_to_history(item)
            self.refresh_history()

    def on_batch_error(self, error_msg: str):
        self.batch_progress.hide()
        self.batch_progress_label.clear()
        self.batch_run_btn.setEnabled(True)
        QMessageBox.critical(self, "Batch error", error_msg)

    def render_batch_results(self, results: list[dict]):
        self.batch_table.setRowCount(len(results))
        for row, result in enumerate(results):
            self.batch_table.setItem(row, 0, QTableWidgetItem(os.path.basename(result.get("video_path", ""))))
            verdict = result.get("customer_verdict", result.get("prediction", "UNKNOWN"))
            verdict_item = QTableWidgetItem(verdict)
            if verdict == "LIKELY_FAKE":
                verdict_item.setForeground(QColor("#ef4444"))
            elif verdict == "LIKELY_REAL":
                verdict_item.setForeground(QColor("#22c55e"))
            else:
                verdict_item.setForeground(QColor("#f59e0b"))
            self.batch_table.setItem(row, 1, verdict_item)
            self.batch_table.setItem(row, 2, QTableWidgetItem(result.get("confidence", "N/A")))
            self.batch_table.setItem(row, 3, QTableWidgetItem(f"{result.get('probability_fake', 0.0) * 100:.1f}%"))
            self.batch_table.setItem(row, 4, QTableWidgetItem(f"{result.get('model_stem', '')} / {result.get('scan_mode', 'accurate')}"))
            self.batch_table.setItem(row, 5, QTableWidgetItem(result.get("reason_summary", "")))

    def export_single_report(self):
        if not self.last_result:
            QMessageBox.information(self, "No result", "Run an analysis first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export HTML report",
            str(Path(BASE_PATH) / "output" / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"),
            "HTML Files (*.html)",
        )
        if not path:
            return
        Path(path).write_text(self.build_single_report_html(self.last_result), encoding="utf-8")
        self.statusBar().showMessage(f"Report saved: {path}")

    def export_batch_report(self):
        if not self.last_batch_results:
            QMessageBox.information(self, "No batch result", "Run a batch analysis first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export batch HTML report",
            str(Path(BASE_PATH) / "output" / f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"),
            "HTML Files (*.html)",
        )
        if not path:
            return
        Path(path).write_text(self.build_batch_report_html(self.last_batch_results), encoding="utf-8")
        self.statusBar().showMessage(f"Batch report saved: {path}")

    def export_history_json(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export history JSON",
            str(Path(BASE_PATH) / "output" / "history_export.json"),
            "JSON Files (*.json)",
        )
        if not path:
            return
        Path(path).write_text(json.dumps(self.history, indent=2, ensure_ascii=False), encoding="utf-8")
        self.statusBar().showMessage(f"History exported: {path}")

    def choose_benchmark_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose benchmark dataset folder", self.benchmark_dir_edit.text())
        if folder:
            self.benchmark_dir_edit.setText(folder)

    def run_benchmark(self):
        self.benchmark_btn.setEnabled(False)
        self.benchmark_progress.show()
        self.benchmark_progress.setRange(0, 0)
        self.benchmark_output.clear()
        self.benchmark_status.setText("Preparing offline benchmark...")

        self.benchmark_thread = BenchmarkThread(
            self.models_dir,
            self.benchmark_dir_edit.text(),
            self.config_path,
            self.benchmark_limit.value(),
        )
        self.benchmark_thread.progress.connect(self.benchmark_status.setText)
        self.benchmark_thread.finished.connect(self.on_benchmark_finished)
        self.benchmark_thread.error.connect(self.on_benchmark_error)
        self.benchmark_thread.start()

    def on_benchmark_finished(self, payload: dict):
        self.benchmark_btn.setEnabled(True)
        self.benchmark_progress.hide()
        self.benchmark_status.setText(f"Benchmark complete. Best model: {payload.get('best_model', 'N/A')}")
        self.benchmark_output.setPlainText(json.dumps(payload, indent=2, ensure_ascii=False))
        self.benchmark_empty.hide()
        Path(self.benchmark_best_path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self.recommended_model = payload.get("best_model")
        self.load_models()

    def on_benchmark_error(self, error_msg: str):
        self.benchmark_btn.setEnabled(True)
        self.benchmark_progress.hide()
        self.benchmark_status.setText("Benchmark failed.")
        QMessageBox.critical(self, "Benchmark error", error_msg)

    def build_single_report_html(self, result: dict):
        reason_points = "".join(f"<li>{point}</li>" for point in result.get("reason_points", []))
        quality_flags = "".join(f"<li>{flag}</li>" for flag in result.get("quality_flags", []))
        explanations = "".join(f"<li>{item}</li>" for item in result.get("explanations", []))
        return f"""
        <html><head><meta charset="utf-8"><title>AI Checker Report</title></head>
        <body style="font-family:Segoe UI;background:#0f172a;color:#e2e8f0;padding:24px;">
            <h1>{result.get('verdict_headline', result.get('prediction', 'Result'))}</h1>
            <p><b>Reason:</b> {result.get('reason_summary', '')}</p>
            <p><b>Video:</b> {result.get('video_path', '')}</p>
            <p><b>Model:</b> {result.get('model_stem', '')} | <b>Mode:</b> {result.get('scan_mode', 'accurate')}</p>
            <p><b>Confidence:</b> {result.get('confidence', 'N/A')} | <b>Fake probability:</b> {result.get('probability_fake', 0.0):.1%}</p>
            <h2>Reason points</h2><ul>{reason_points or '<li>No extra points</li>'}</ul>
            <h2>Quality flags</h2><ul>{quality_flags or '<li>No warnings</li>'}</ul>
            <h2>Explanations</h2><ul>{explanations or '<li>No explanations</li>'}</ul>
            <p style="margin-top:24px;color:#94a3b8;">Generated at {datetime.now().isoformat()}</p>
        </body></html>
        """

    def build_batch_report_html(self, results: list[dict]):
        rows = []
        for item in results:
            rows.append(
                "<tr>"
                f"<td>{os.path.basename(item.get('video_path', ''))}</td>"
                f"<td>{item.get('customer_verdict', item.get('prediction', 'UNKNOWN'))}</td>"
                f"<td>{item.get('confidence', 'N/A')}</td>"
                f"<td>{item.get('probability_fake', 0.0):.1%}</td>"
                f"<td>{item.get('model_stem', '')}</td>"
                f"<td>{item.get('scan_mode', 'accurate')}</td>"
                f"<td>{item.get('reason_summary', '')}</td>"
                "</tr>"
            )
        return f"""
        <html><head><meta charset="utf-8"><title>Batch Report</title></head>
        <body style="font-family:Segoe UI;background:#0f172a;color:#e2e8f0;padding:24px;">
            <h1>Batch Analysis Report</h1>
            <p><b>Total videos:</b> {len(results)}</p>
            <table border="1" cellpadding="8" cellspacing="0" style="border-collapse:collapse;background:#152238;">
                <tr><th>Video</th><th>Verdict</th><th>Confidence</th><th>Fake %</th><th>Model</th><th>Mode</th><th>Summary</th></tr>
                {''.join(rows)}
            </table>
            <p style="margin-top:24px;color:#94a3b8;">Generated at {datetime.now().isoformat()}</p>
        </body></html>
        """

    def load_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r", encoding="utf-8") as file:
                    return json.load(file)
            except Exception:
                return []
        return []

    def save_to_history(self, result: dict):
        entry = {
            "timestamp": result.get("timestamp", datetime.now().isoformat()),
            "video_path": result.get("video_path", ""),
            "prediction": result.get("prediction", "UNKNOWN"),
            "customer_verdict": result.get("customer_verdict", result.get("prediction", "UNKNOWN")),
            "probability_fake": result.get("probability_fake", 0.0),
            "model_stem": result.get("model_stem", ""),
            "scan_mode": result.get("scan_mode", "accurate"),
            "confidence": result.get("confidence", result.get("fusion_result", {}).get("confidence", "N/A")),
        }
        self.history.insert(0, entry)
        self.history = self.history[:400]
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        with open(self.history_file, "w", encoding="utf-8") as file:
            json.dump(self.history, file, indent=2, ensure_ascii=False)

    def refresh_history(self):
        self.history = self.load_history()
        search = self.history_search.text().strip().lower() if hasattr(self, "history_search") else ""
        verdict_filter = self.history_filter.currentText() if hasattr(self, "history_filter") else "All verdicts"

        filtered = []
        for entry in self.history:
            video_name = os.path.basename(entry.get("video_path", "")).lower()
            model_name = entry.get("model_stem", "").lower()
            verdict = entry.get("customer_verdict", entry.get("prediction", ""))
            if search and search not in video_name and search not in model_name:
                continue
            if verdict_filter != "All verdicts" and verdict != verdict_filter:
                continue
            filtered.append(entry)

        self.history_table.setRowCount(len(filtered))
        for row, entry in enumerate(filtered):
            try:
                ts = datetime.fromisoformat(entry["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                ts = entry.get("timestamp", "")
            self.history_table.setItem(row, 0, QTableWidgetItem(ts))
            self.history_table.setItem(row, 1, QTableWidgetItem(os.path.basename(entry.get("video_path", ""))))
            self.history_table.setItem(row, 2, QTableWidgetItem(f"{entry.get('model_stem', '')} / {entry.get('scan_mode', 'accurate')}"))
            verdict = entry.get("customer_verdict", entry.get("prediction", "UNKNOWN"))
            verdict_item = QTableWidgetItem(verdict)
            if verdict == "LIKELY_FAKE":
                verdict_item.setForeground(QColor("#ef4444"))
            elif verdict == "LIKELY_REAL":
                verdict_item.setForeground(QColor("#22c55e"))
            else:
                verdict_item.setForeground(QColor("#f59e0b"))
            self.history_table.setItem(row, 3, verdict_item)
            self.history_table.setItem(row, 4, QTableWidgetItem(entry.get("confidence", "N/A")))
        self.history_empty.setVisible(len(filtered) == 0)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
