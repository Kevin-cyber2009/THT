#!/usr/bin/env python3
# server.py - Async với Threading (không cần Celery/Redis, chạy được free tier)

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tempfile
import os
from pathlib import Path
import logging
import time
import uuid
import threading

from src.utils import load_config
from src.classifier import VideoClassifier
from src.features import FeatureExtractor
from src.fusion import ScoreFusion

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH    = os.environ.get('MODEL_PATH',  'models/alpha.pkl')
CONFIG_PATH   = os.environ.get('CONFIG_PATH', 'config.yaml')
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
start_time    = time.time()

# ─────────────────────────────────────────────
# In-memory job store (thread-safe)
# ─────────────────────────────────────────────
_jobs_lock = threading.Lock()
_jobs: dict = {}   # { job_id: { status, result, error, updated_at } }

def set_job(job_id: str, data: dict):
    with _jobs_lock:
        _jobs[job_id] = {**data, "updated_at": time.time()}

def get_job(job_id: str):
    with _jobs_lock:
        return _jobs.get(job_id)

def cleanup_old_jobs():
    """Xóa jobs cũ hơn 1 giờ để tránh memory leak"""
    cutoff = time.time() - 3600
    with _jobs_lock:
        to_delete = [k for k, v in _jobs.items() if v.get("updated_at", 0) < cutoff]
        for k in to_delete:
            _jobs.pop(k, None)


# ─────────────────────────────────────────────
# Models (load 1 lần duy nhất)
# ─────────────────────────────────────────────
classifier        = None
feature_extractor = None
fusion_engine     = None

def initialize_models():
    global classifier, feature_extractor, fusion_engine
    logger.info("🔄 Loading models...")
    try:
        config = load_config(CONFIG_PATH)
        classifier = VideoClassifier(config)
        classifier.load(MODEL_PATH)
        logger.info(f"✅ Classifier loaded from {MODEL_PATH}")
        feature_extractor = FeatureExtractor(config)
        logger.info("✅ Feature extractor initialized")
        fusion_engine = ScoreFusion(config)
        logger.info("✅ Fusion engine initialized")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        import traceback; traceback.print_exc()
        return False


# ─────────────────────────────────────────────
# Background worker (chạy trong thread riêng)
# ─────────────────────────────────────────────
def run_analysis(job_id: str, video_path: str, original_filename: str):
    try:
        set_job(job_id, {"status": "processing", "step": "extracting_features"})
        logger.info(f"[{job_id}] 🔍 Extracting: {original_filename}")
        features_dict, metadata = feature_extractor.extract_from_video(video_path)

        set_job(job_id, {"status": "processing", "step": "predicting"})
        logger.info(f"[{job_id}] 🤖 Predicting...")
        feature_names  = classifier.feature_names or feature_extractor.get_feature_names()
        feature_vector = feature_extractor.features_to_vector(features_dict, feature_names)
        X = feature_vector.reshape(1, -1)
        pred, prob = classifier.predict(X)

        set_job(job_id, {"status": "processing", "step": "fusion"})
        logger.info(f"[{job_id}] 🔗 Fusing...")
        artifact_score = fusion_engine.compute_artifact_score(features_dict)
        reality_score  = fusion_engine.compute_reality_score(features_dict)
        fusion_result  = fusion_engine.fuse_scores(artifact_score, reality_score, 0.5)
        explanations   = fusion_engine.generate_explanation(features_dict, fusion_result)

        result = {
            "prediction":       "FAKE" if pred[0] == 1 else "REAL",
            "probability_fake": float(prob[0]),
            "probability_real": float(1 - prob[0]),
            "confidence":       fusion_result["confidence"],
            "artifact_score":   float(artifact_score),
            "reality_score":    float(reality_score),
            "explanations":     explanations,
            "metadata": {
                "filename":   original_filename,
                "num_frames": metadata.get("num_frames", 0),
                "duration":   metadata.get("duration", 0),
                "fps":        metadata.get("fps", 0),
            }
        }

        set_job(job_id, {"status": "done", "result": result})
        logger.info(f"[{job_id}] ✅ {result['prediction']} ({result['probability_fake']:.1%})")

    except Exception as e:
        logger.error(f"[{job_id}] ❌ {e}")
        set_job(job_id, {"status": "error", "error": str(e)})

    finally:
        try:
            if os.path.exists(video_path):
                os.unlink(video_path)
        except Exception:
            pass
        cleanup_old_jobs()


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route('/')
def home():
    return jsonify({
        'name':    'Deepfake Detector API',
        'version': '2.0.0',
        'mode':    'async (threading)',
        'endpoints': {
            'health':  'GET  /health',
            'analyze': 'POST /api/analyze',
            'result':  'GET  /api/result/<job_id>',
            'stats':   'GET  /api/stats',
        }
    })


@app.route('/health')
def health_check():
    return jsonify({
        'status':         'healthy',
        'model_loaded':   classifier is not None,
        'active_jobs':    sum(1 for j in _jobs.values() if j.get('status') == 'processing'),
        'uptime_seconds': round(time.time() - start_time, 2),
        'version':        '2.0.0',
    })


@app.route('/ping')
def ping():
    return 'pong', 200


@app.route('/api/analyze', methods=['POST'])
def analyze_video():
    """Upload video → trả job_id NGAY → xử lý nền trong thread."""
    if classifier is None:
        return jsonify({'success': False, 'error': 'Models not loaded'}), 500

    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'No video file provided'}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'success': False, 'error': 'Empty filename'}), 400

    video_file.seek(0, os.SEEK_END)
    file_size = video_file.tell()
    video_file.seek(0)

    if file_size > MAX_FILE_SIZE:
        return jsonify({'success': False,
                        'error': f'File too large (max {MAX_FILE_SIZE // 1024 // 1024}MB)'}), 400

    logger.info(f"📥 Received: {video_file.filename} ({file_size / 1024 / 1024:.1f}MB)")

    # Lưu file tạm — thread sẽ tự xóa sau khi xong
    suffix = Path(video_file.filename).suffix or '.mp4'
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    video_file.save(tmp.name)
    tmp.close()

    # Tạo job + chạy thread nền
    job_id = str(uuid.uuid4())
    set_job(job_id, {"status": "queued"})

    t = threading.Thread(
        target=run_analysis,
        args=(job_id, tmp.name, video_file.filename),
        daemon=True
    )
    t.start()

    return jsonify({
        'success':  True,
        'job_id':   job_id,
        'status':   'queued',
        'poll_url': f'/api/result/{job_id}',
    }), 202


@app.route('/api/result/<job_id>', methods=['GET'])
def get_result(job_id: str):
    job = get_job(job_id)

    if job is None:
        return jsonify({
            'job_id': job_id,
            'status': 'not_found',
            'error':  'Job không tồn tại hoặc đã hết hạn (>1 giờ)'
        }), 404

    status = job.get('status')

    if status == 'queued':
        return jsonify({'job_id': job_id, 'status': 'pending', 'message': 'Đang chờ xử lý...'})

    elif status == 'processing':
        return jsonify({'job_id': job_id, 'status': 'processing', 'step': job.get('step', '')})

    elif status == 'done':
        return jsonify({'job_id': job_id, 'status': 'done', 'success': True, **job['result']})

    elif status == 'error':
        return jsonify({
            'job_id':  job_id,
            'status':  'error',
            'success': False,
            'error':   job.get('error')
        }), 500

    return jsonify({'job_id': job_id, 'status': status})


@app.route('/api/stats')
def get_stats():
    with _jobs_lock:
        jobs_snapshot = dict(_jobs)

    return jsonify({
        'status':         'running',
        'uptime_seconds': round(time.time() - start_time, 2),
        'model_loaded':   classifier is not None,
        'jobs': {
            'total':   len(jobs_snapshot),
            'active':  sum(1 for j in jobs_snapshot.values() if j.get('status') == 'processing'),
            'queued':  sum(1 for j in jobs_snapshot.values() if j.get('status') == 'queued'),
            'done':    sum(1 for j in jobs_snapshot.values() if j.get('status') == 'done'),
            'errors':  sum(1 for j in jobs_snapshot.values() if j.get('status') == 'error'),
        }
    })


# ─────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────
initialize_models()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🚀 Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)