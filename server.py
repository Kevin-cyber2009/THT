#!/usr/bin/env python3
# server.py - Optimized for Render with keep-alive endpoint

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tempfile
import os
from pathlib import Path
import logging
import time

# Import your detector modules
from src.utils import load_config
from src.classifier import VideoClassifier
from src.features import FeatureExtractor
from src.fusion import ScoreFusion

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
classifier = None
feature_extractor = None
fusion_engine = None
start_time = time.time()

# Configuration
MODEL_PATH = os.environ.get('MODEL_PATH', 'models/alpha.pkl')
CONFIG_PATH = os.environ.get('CONFIG_PATH', 'config.yaml')
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB


def initialize_models():
    """Initialize models at startup"""
    global classifier, feature_extractor, fusion_engine
    
    logger.info("🔄 Loading models...")
    
    try:
        config = load_config(CONFIG_PATH)
        
        # Load classifier
        classifier = VideoClassifier(config)
        classifier.load(MODEL_PATH)
        logger.info(f"✅ Classifier loaded from {MODEL_PATH}")
        
        # Initialize feature extractor
        feature_extractor = FeatureExtractor(config)
        logger.info("✅ Feature extractor initialized")
        
        # Initialize fusion engine
        fusion_engine = ScoreFusion(config)
        logger.info("✅ Fusion engine initialized")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        return False


@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        'name': 'Deepfake Detector API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            'health': '/health',
            'analyze': '/api/analyze',
            'stats': '/api/stats'
        }
    })


@app.route('/health')
def health_check():
    """Health check endpoint - for keep-alive pings"""
    uptime = time.time() - start_time
    return jsonify({
        'status': 'healthy',
        'model_loaded': classifier is not None,
        'uptime_seconds': round(uptime, 2),
        'version': '1.0.0',
        'timestamp': time.time()
    })


@app.route('/ping')
def ping():
    """Alternative ping endpoint"""
    return 'pong', 200


@app.route('/api/analyze', methods=['POST'])
def analyze_video():
    """Main analysis endpoint"""
    try:
        # Check if models loaded
        if classifier is None:
            return jsonify({
                'success': False,
                'error': 'Models not loaded'
            }), 500
        
        # Check if video file in request
        if 'video' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No video file provided'
            }), 400
        
        video_file = request.files['video']
        
        if video_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Empty filename'
            }), 400
        
        # Check file size
        video_file.seek(0, os.SEEK_END)
        file_size = video_file.tell()
        video_file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            return jsonify({
                'success': False,
                'error': f'File too large (max {MAX_FILE_SIZE // 1024 // 1024}MB)'
            }), 400
        
        logger.info(f"📥 Received video: {video_file.filename} ({file_size / 1024 / 1024:.1f}MB)")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            video_file.save(tmp_file.name)
            temp_video_path = tmp_file.name
        
        try:
            # Extract features
            logger.info("🔍 Extracting features...")
            features_dict, metadata = feature_extractor.extract_from_video(temp_video_path)
            
            # Convert to vector
            feature_names = classifier.feature_names or feature_extractor.get_feature_names()
            feature_vector = feature_extractor.features_to_vector(features_dict, feature_names)
            X = feature_vector.reshape(1, -1)
            
            # Predict
            logger.info("🤖 Making prediction...")
            pred, prob = classifier.predict(X)
            
            # Compute component scores
            artifact_score = fusion_engine.compute_artifact_score(features_dict)
            reality_score = fusion_engine.compute_reality_score(features_dict)
            
            # Fusion
            fusion_result = fusion_engine.fuse_scores(
                artifact_score,
                reality_score,
                0.5  # Default stress score
            )
            
            # Generate explanations
            explanations = fusion_engine.generate_explanation(features_dict, fusion_result)
            
            # Prepare response
            response = {
                'success': True,
                'prediction': 'FAKE' if pred[0] == 1 else 'REAL',
                'probability_fake': float(prob[0]),
                'probability_real': float(1 - prob[0]),
                'confidence': fusion_result['confidence'],
                'artifact_score': float(artifact_score),
                'reality_score': float(reality_score),
                'explanations': explanations,
                'metadata': {
                    'num_frames': metadata.get('num_frames', 0),
                    'duration': metadata.get('duration', 0),
                    'fps': metadata.get('fps', 0)
                }
            }
            
            logger.info(f"✅ Analysis complete: {response['prediction']} ({response['probability_fake']:.1%})")
            
            return jsonify(response)
        
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_video_path)
            except:
                pass
    
    except Exception as e:
        logger.error(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/stats')
def get_stats():
    """Get server statistics"""
    uptime = time.time() - start_time
    return jsonify({
        'model_path': MODEL_PATH,
        'config_path': CONFIG_PATH,
        'max_file_size_mb': MAX_FILE_SIZE // 1024 // 1024,
        'uptime_seconds': round(uptime, 2),
        'status': 'running',
        'model_loaded': classifier is not None
    })


if __name__ == '__main__':
    # Initialize models on startup
    if not initialize_models():
        print("❌ Failed to initialize models. Exiting.")
        exit(1)
    
    # Get port from environment
    port = int(os.environ.get('PORT', 5000))
    
    print("\n" + "="*60)
    print("🚀 DEEPFAKE DETECTOR API SERVER")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Config: {CONFIG_PATH}")
    print(f"Port: {port}")
    print(f"Max file size: {MAX_FILE_SIZE // 1024 // 1024}MB")
    print("="*60)
    print("\nEndpoints:")
    print("  GET  /              - API info")
    print("  GET  /health        - Health check")
    print("  GET  /ping          - Simple ping")
    print("  POST /api/analyze   - Analyze video")
    print("  GET  /api/stats     - Server stats")
    print("\nServer starting...")
    print("="*60 + "\n")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=port, debug=False)
