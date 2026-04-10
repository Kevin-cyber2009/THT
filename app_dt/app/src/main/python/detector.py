"""
Enhanced Detector for AI Video Detection
========================================

Integrates:
- Traditional forensic features (FFT, DCT, PRNU, Optical Flow)
- Deep learning features (ResNet50, EfficientNet)
- Face analysis features (eye, symmetry, skin texture)
- Temporal features (motion, frequency, noise)
- Ensemble model inference
"""

import json
import os
import sys
import types
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import cv2


class _FakeLGBM:
    def __init__(self, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__.update(state)

    def predict_proba(self, x):
        raise RuntimeError("Use ONNX instead")

    def predict(self, x):
        raise RuntimeError("Use ONNX instead")

    @property
    def feature_importances_(self):
        return []


class _AutoMock(types.ModuleType):
    def __getattr__(self, name):
        return _FakeLGBM


def _make_lgb_module(name):
    module = _AutoMock(name)
    module.LGBMClassifier = _FakeLGBM
    module.LGBMRegressor = _FakeLGBM
    module.LGBMRanker = _FakeLGBM
    module.LGBMModel = _FakeLGBM
    module.Dataset = type('Dataset', (), {'__init__': lambda self, *args, **kwargs: None})
    module.train = lambda *args, **kwargs: None
    module.cv = lambda *args, **kwargs: None
    return module


for _mod in [
    'lightgbm',
    'lightgbm.sklearn',
    'lightgbm.basic',
    'lightgbm.compat',
    'lightgbm.callback',
    'lightgbm.engine',
    'lightgbm.plotting',
    'lightgbm.dask',
]:
    sys.modules[_mod] = _make_lgb_module(_mod)


BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from src.features import FeatureExtractor
from src.fusion import ScoreFusion
from src.utils import load_config

try:
    from src.face_analyzer import FaceAnalyzer
    FACE_ANALYZER_AVAILABLE = True
except ImportError:
    FACE_ANALYZER_AVAILABLE = False
    FaceAnalyzer = None

try:
    from src.temporal_features import TemporalAnalyzer
    TEMPORAL_ANALYZER_AVAILABLE = True
except ImportError:
    TEMPORAL_ANALYZER_AVAILABLE = False
    TemporalAnalyzer = None


_scaler = None
_feature_names = None
_extractor = None
_fusion = None
_config = None
_n_features = 0
_onnx_sessions = {}
_onnx_models = []
_scaler_mean = None
_scaler_scale = None
_face_analyzer = None
_temporal_analyzer = None
_traditional_feature_count = 28

_DEEP_NAMES = [
    'deep_feat_mean',
    'deep_feat_std',
    'deep_feat_max',
    'deep_feat_min',
    'deep_temporal_var_mean',
    'deep_temporal_var_std',
    'deep_l2_norm_mean',
    'deep_l2_norm_std',
    'deep_similarity_mean',
    'deep_similarity_std',
    'deep_sparsity',
]
_DEEP_NAMES_SET = frozenset(_DEEP_NAMES)

_FACE_FEATURES = [
    'eye_aspect_ratio_mean',
    'face_symmetry_score',
    'skin_texture_variance',
    'blink_rate',
    'landmark_temporal_variance',
    'lip_irregularity',
]

_TEMPORAL_FEATURES = [
    'frame_diff_mean',
    'flow_magnitude_mean',
    'motion_smoothness',
    'temporal_fft_ratio',
    'noise_temporal_variance',
]

_SCAN_PROFILES = {
    'quick': {
        'label': 'Quét nhanh',
        'fps': 6,
        'max_frames': 48,
        'sample_frames': 8,
    },
    'accurate': {
        'label': 'Quét chính xác',
        'fps': 6,
        'max_frames': 200,
        'sample_frames': 20,
    },
}


def load_models(scaler_path: str, config_path: str) -> str:
    global _scaler, _feature_names, _extractor, _fusion, _config
    global _n_features, _onnx_sessions, _onnx_models, _scaler_mean, _scaler_scale
    global _face_analyzer, _temporal_analyzer

    try:
        _config = load_config(config_path)

        preproc = _config.setdefault('preprocessing', {})
        preproc.setdefault('resize_width', 512)
        preproc.setdefault('resize_height', 288)
        preproc.setdefault('fps', 6)
        preproc.setdefault('max_frames', 1000)

        models_dir = os.path.dirname(os.path.abspath(scaler_path))
        _config.setdefault('deep_learning', {})['models_dir'] = models_dir
        _config.setdefault('features', {})['use_deep_features'] = False

        import joblib

        data = joblib.load(scaler_path)
        if 'scaler' not in data:
            return 'LỖI: tệp pkl không chứa khóa "scaler"'

        _scaler = data['scaler']
        _feature_names = data.get('feature_names')
        _n_features = int(_scaler.n_features_in_) if hasattr(_scaler, 'n_features_in_') else int(data.get('n_features', 39))

        if hasattr(_scaler, 'mean_'):
            _scaler_mean = np.array(_scaler.mean_, dtype=np.float64)
            _scaler_scale = np.array(_scaler.scale_, dtype=np.float64)
        else:
            _scaler_mean = None
            _scaler_scale = None

        print(f"[detector] n_features={_n_features}")

        _extractor = FeatureExtractor(_config)
        _fusion = ScoreFusion(_config)

        if FACE_ANALYZER_AVAILABLE:
            _face_analyzer = FaceAnalyzer(_config)
            print("[detector] FaceAnalyzer initialized")

        if TEMPORAL_ANALYZER_AVAILABLE:
            _temporal_analyzer = TemporalAnalyzer(_config)
            print("[detector] TemporalAnalyzer initialized")

        scaler_stem = os.path.splitext(os.path.basename(scaler_path))[0]
        if scaler_stem.endswith('_scaler'):
            scaler_stem = scaler_stem[:-len('_scaler')]

        import onnxruntime as rt

        available_models = []
        for fname in os.listdir(models_dir):
            if fname.endswith('.onnx') and not fname.endswith('_features.onnx'):
                available_models.append(fname)

        print(f"[detector] Found ONNX models: {available_models}")

        preferred_model = f"{scaler_stem}.onnx"
        model_priority = [preferred_model] + [m for m in available_models if m != preferred_model]

        _onnx_sessions = {}
        _onnx_models = []

        for model_name in model_priority:
            model_path = os.path.join(models_dir, model_name)
            if os.path.exists(model_path):
                try:
                    session = rt.InferenceSession(model_path)
                    _onnx_sessions[model_name] = session
                    _onnx_models.append(model_name)
                    print(f"[detector] Loaded ONNX: {model_name}")
                except Exception as e:
                    print(f"[detector] Failed to load {model_name}: {e}")

        if not _onnx_models:
            print("[detector] WARNING: No ONNX models loaded, will use fusion only")

        print(f"[detector] Ready with {len(_onnx_models)} ONNX models")
        return 'OK'

    except Exception as exc:
        import traceback
        return f'LỖI: {exc}\n{traceback.format_exc()}'


def _manual_scale(vector: np.ndarray) -> np.ndarray:
    if _scaler_mean is None or _scaler_scale is None:
        return vector
    
    vector = vector.astype(np.float64)
    scaled = (vector - _scaler_mean) / (_scaler_scale + 1e-10)
    scaled = np.clip(scaled, -10.0, 10.0)
    return scaled.astype(np.float32)


def _neutral_value_for(idx: int) -> float:
    if _scaler_mean is not None and idx < len(_scaler_mean):
        return float(_scaler_mean[idx])
    if _scaler is not None and hasattr(_scaler, 'mean_') and idx < len(_scaler.mean_):
        return float(_scaler.mean_[idx])
    return 0.0


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _assess_input_quality(metadata: dict) -> dict:
    frame_count = int(metadata.get('num_frames', 0) or 0)
    width = int(metadata.get('width', 0) or 0)
    height = int(metadata.get('height', 0) or 0)
    duration = float(metadata.get('duration', 0.0) or 0.0)
    fps = float(metadata.get('fps', metadata.get('sampling_fps', 0.0)) or 0.0)

    score = 1.0
    flags = []

    if frame_count < 16:
        score -= 0.35
        flags.append('Số khung hình được lấy mẫu quá ít')
    elif frame_count < 30:
        score -= 0.15
        flags.append('Số khung hình được lấy mẫu còn hạn chế')

    if duration and duration < 2.0:
        score -= 0.30
        flags.append('Video quá ngắn')
    elif duration and duration < 4.0:
        score -= 0.12
        flags.append('Thời lượng video hơi ngắn')

    if width and height and width * height < 640 * 360:
        score -= 0.20
        flags.append('Độ phân giải nguồn thấp')
    elif width and height and width * height > 1920 * 1080:
        score += 0.05

    if fps and fps < 12:
        score -= 0.10
        flags.append('FPS nguồn thấp')

    score = _clip01(score)
    if score >= 0.85:
        label = 'HIGH'
    elif score >= 0.60:
        label = 'MEDIUM'
    else:
        label = 'LOW'

    return {
        'score': score,
        'label': label,
        'flags': flags[:4],
    }


def _resolve_scan_mode(scan_mode: str) -> str:
    normalized = (scan_mode or 'accurate').strip().lower()
    return normalized if normalized in _SCAN_PROFILES else 'accurate'


def _apply_scan_profile(config: dict, scan_mode: str) -> dict:
    runtime = json.loads(json.dumps(config))
    profile = _SCAN_PROFILES[_resolve_scan_mode(scan_mode)]
    runtime.setdefault('preprocessing', {}).update({
        'fps': profile['fps'],
        'max_frames': profile['max_frames'],
    })
    runtime.setdefault('deep_learning', {})['sample_frames'] = profile['sample_frames']
    return runtime


def _build_reason_points(
    artifact: float,
    reality: float,
    stress: float,
    model_prob: float,
    fusion_prob: float,
    quality: dict,
) -> List[str]:
    points = []

    if quality['score'] < 0.50 and quality['flags']:
        points.append('video đầu vào chưa đủ rõ hoặc quá ngắn')
    if artifact >= 0.65:
        points.append('tín hiệu artifact và texture bất thường rõ rệt')
    if stress >= 0.60:
        points.append('chuyển động giữa các khung hình kém ổn định')
    if model_prob >= 0.70 and fusion_prob >= 0.60:
        points.append('cả model và fusion đều nghiêng về video AI')
    if reality >= 0.65:
        points.append('chỉ số reality vẫn giữ được độ tự nhiên tốt')
    if model_prob <= 0.30 and fusion_prob <= 0.40:
        points.append('cả model và fusion đều nghiêng về video thật')

    if not points and quality['flags']:
        points.append('cần cân nhắc chất lượng nguồn video trước khi kết luận')
    if not points:
        points.append('các tín hiệu hiện tại chưa quá cực đoan')

    return points[:3]


def _determine_customer_verdict(blended_prob: float, confidence: str, quality: dict) -> Tuple[str, str]:
    if quality['score'] < 0.50:
        return 'INSUFFICIENT_QUALITY', 'Không đủ chất lượng để kết luận'
    if 0.40 <= blended_prob <= 0.60 or confidence == 'LOW':
        return 'UNCERTAIN', 'Cần kiểm tra thêm'
    if blended_prob >= 0.5:
        return 'LIKELY_FAKE', 'Có khả năng là video AI'
    return 'LIKELY_REAL', 'Có khả năng là video thật'


def _build_reason_summary(headline: str, reasons: List[str]) -> str:
    if not reasons:
        return headline
    return f"{headline}, vì {reasons[0]}."


def get_deep_sample_plan(video_path: str, scan_mode: str = 'accurate') -> str:
    if _config is None:
        return json.dumps({'error': 'Mô hình chưa được tải - hãy gọi load_models() trước'})

    try:
        runtime_config = _apply_scan_profile(_config, scan_mode)
        resolved_scan_mode = _resolve_scan_mode(scan_mode)
        sample_frames = int(_SCAN_PROFILES[resolved_scan_mode]['sample_frames'])
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return json.dumps({'times_us': []})

        original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()

        if original_fps <= 0:
            original_fps = 30.0

        preproc = runtime_config.get('preprocessing', {})
        target_fps = float(preproc.get('fps', 6) or 6)
        max_frames = int(preproc.get('max_frames', 200) or 200)
        frame_interval = max(1, int(original_fps / target_fps))

        extracted_indices = list(range(0, total_frames, frame_interval))[:max_frames]
        if not extracted_indices:
            return json.dumps({'times_us': []})

        if len(extracted_indices) > sample_frames:
            sample_positions = np.linspace(0, len(extracted_indices) - 1, sample_frames).astype(int)
            chosen_indices = [extracted_indices[i] for i in sample_positions]
        else:
            chosen_indices = extracted_indices

        times_us = [int((frame_idx / original_fps) * 1_000_000) for frame_idx in chosen_indices]
        return json.dumps({
            'times_us': times_us,
            'frame_indices': chosen_indices,
            'original_fps': original_fps,
            'frame_interval': frame_interval,
            'scan_mode': resolved_scan_mode,
            'total_extracted': len(extracted_indices),
        })
    except Exception as exc:
        return json.dumps({'error': str(exc), 'times_us': []})


def _run_ensemble_inference(scaled_vector: np.ndarray) -> Tuple[int, float, List[Dict]]:
    if not _onnx_sessions:
        return 0, 0.5, []
    
    results = []
    for model_name, session in _onnx_sessions.items():
        try:
            input_name = session.get_inputs()[0].name
            X = scaled_vector.astype(np.float32).reshape(1, -1)
            outputs = session.run(None, {input_name: X})
            
            label = 0
            prob_fake = 0.5
            
            if len(outputs) >= 1:
                label_out = outputs[0]
                if hasattr(label_out, 'flatten'):
                    label = int(label_out.flatten()[0])
                elif isinstance(label_out, (list, tuple)):
                    label = int(label_out[0])
            
            if len(outputs) >= 2:
                prob_out = outputs[1]
                if isinstance(prob_out, np.ndarray):
                    flat = prob_out.flatten()
                    if len(flat) >= 2:
                        prob_fake = float(flat[1])
                    elif len(flat) == 1:
                        prob_fake = float(flat[0])
                elif isinstance(prob_out, list) and len(prob_out) > 0:
                    prob_fake = float(prob_out[0]) if isinstance(prob_out[0], (int, float)) else 0.5
            
            results.append({
                'model': model_name,
                'label': label,
                'prob_fake': prob_fake,
            })
        except Exception as e:
            print(f"[detector] Model {model_name} inference failed: {e}")
    
    if not results:
        return 0, 0.5, []
    
    if len(results) == 1:
        r = results[0]
        return r['label'], r['prob_fake'], results
    
    probs = [r['prob_fake'] for r in results]
    labels = [r['label'] for r in results]
    
    avg_prob = float(np.mean(probs))
    std_prob = float(np.std(probs))
    
    confidence_weight = 1.0 / (1.0 + std_prob * 2)
    final_prob = avg_prob * confidence_weight + 0.5 * (1 - confidence_weight)
    
    final_label = 1 if avg_prob >= 0.5 else 0
    
    for r in results:
        r['weight'] = 1.0 / len(results)
    
    print(f"[detector] Ensemble: avg_prob={avg_prob:.4f}, std={std_prob:.4f}, final={final_prob:.4f}")
    
    return final_label, final_prob, results


def extract_features(video_path: str, deep_json: str = '{}', scan_mode: str = 'accurate') -> str:
    global _face_analyzer, _temporal_analyzer
    
    if _scaler is None or _extractor is None:
        return json.dumps({'error': 'Mô hình chưa được tải - hãy gọi load_models() trước'})

    try:
        runtime_config = _apply_scan_profile(_config, scan_mode)
        runtime_extractor = FeatureExtractor(runtime_config)
        runtime_fusion = ScoreFusion(runtime_config)
        resolved_scan_mode = _resolve_scan_mode(scan_mode)
        scan_profile = _SCAN_PROFILES[resolved_scan_mode]

        features, metadata = runtime_extractor.extract_from_video(video_path)
        deep_features_valid = False
        deep_features_count = 0
        face_features_count = 0
        temporal_features_count = 0

        deep_injected = {}
        if deep_json and deep_json not in ('{}', '', 'null', '[]'):
            try:
                parsed = json.loads(deep_json)
                if isinstance(parsed, dict):
                    deep_injected = {k: float(v) for k, v in parsed.items() if k in _DEEP_NAMES_SET}
                elif isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
                    for item in parsed:
                        for k, v in item.items():
                            if k in _DEEP_NAMES_SET:
                                deep_injected[k] = float(v)
            except Exception as exc:
                print(f"[detector] deep_json parse error: {exc}")

        if deep_injected:
            features.update(deep_injected)
            deep_features_valid = True
            deep_features_count = len(deep_injected)
            print(f"[detector] Injected {deep_features_count} deep features")

        if _face_analyzer is not None and _temporal_analyzer is not None:
            try:
                frames, _ = runtime_extractor.preprocessor.preprocess(video_path)
                
                face_feats = _face_analyzer.extract_all_features(frames.tolist() if hasattr(frames, 'tolist') else frames)
                for key, value in face_feats.items():
                    if key.startswith(('eye_', 'face_symmetry', 'skin_', 'blink_', 'landmark_', 'lip_', 'nose_')):
                        features[f'face_{key}'] = value
                        face_features_count += 1
                
                temporal_feats = _temporal_analyzer.extract_all_features(frames)
                for key, value in temporal_feats.items():
                    features[f'temporal_{key}'] = value
                    temporal_features_count += 1
                
                print(f"[detector] Face features: {face_features_count}, Temporal features: {temporal_features_count}")
                
            except Exception as e:
                print(f"[detector] Face/Temporal analysis failed: {e}")

        for name in _DEEP_NAMES:
            if name not in features:
                features[name] = 0.0

        vector = []
        missing = []
        for i, name in enumerate(_feature_names):
            value = features.get(name)
            if value is None:
                missing.append(name)
                value = _neutral_value_for(i)
            if np.isnan(value) or np.isinf(value):
                value = _neutral_value_for(i)
            vector.append(float(value))

        vector = np.array(vector, dtype=np.float64)
        
        if len(vector) < _n_features:
            pad = np.array([_neutral_value_for(i) for i in range(len(vector), _n_features)])
            vector = np.concatenate([vector, pad])
        elif len(vector) > _n_features:
            vector = vector[:_n_features]

        print(f"[detector] Vector: shape={vector.shape}, mean={np.mean(vector):.4f}")

        scaled = _manual_scale(vector)
        print(f"[detector] Scaled: mean={np.mean(scaled):.4f}, std={np.std(scaled):.4f}")

        artifact = runtime_fusion.compute_artifact_score(features)
        reality = runtime_fusion.compute_reality_score(features)
        stress_proxy = runtime_fusion.compute_stress_proxy(features)
        fusion = runtime_fusion.fuse_scores(artifact, reality, stress_proxy)
        explanations = runtime_fusion.generate_explanation(features, fusion)
        input_quality = _assess_input_quality(metadata)

        model_label = 0
        model_prob = fusion['final_probability']
        model_used = False
        ensemble_details = []

        if _onnx_sessions:
            model_label, model_prob, ensemble_details = _run_ensemble_inference(scaled)
            model_used = True
            print(f"[detector] Ensemble prediction: label={model_label}, prob={model_prob:.4f}")
        else:
            model_label = 1 if fusion['final_probability'] >= 0.5 else 0
            model_prob = fusion['final_probability']
            print(f"[detector] Using fusion (no ONNX): label={model_label}, prob={model_prob:.4f}")

        quality_adjusted_prob = model_prob
        if input_quality['score'] < 0.60:
            blend_weight = 0.4
            quality_adjusted_prob = model_prob * (1 - blend_weight) + fusion['final_probability'] * blend_weight
            print(f"[detector] Quality adjustment: {model_prob:.4f} -> {quality_adjusted_prob:.4f}")

        confidence_margin = abs(quality_adjusted_prob - 0.5) * 2
        if confidence_margin < 0.20 and input_quality['score'] < 0.70:
            final_prob = 0.5
            print(f"[detector] Low confidence + low quality -> UNCERTAIN")
        else:
            final_prob = quality_adjusted_prob

        reason_points = _build_reason_points(
            artifact,
            reality,
            stress_proxy,
            model_prob,
            fusion['final_probability'],
            input_quality,
        )

        customer_verdict, verdict_headline = _determine_customer_verdict(
            final_prob,
            fusion['confidence'],
            input_quality,
        )
        reason_summary = _build_reason_summary(verdict_headline, reason_points)

        if input_quality['flags']:
            explanations.append(f"Chất lượng đầu vào {input_quality['label']}: " + "; ".join(input_quality['flags']))
        else:
            explanations.append(f"Chất lượng đầu vào {input_quality['label']}: dữ liệu video đủ tốt để đánh giá")

        scaler_mean_list = _scaler_mean.tolist() if _scaler_mean is not None else []
        scaler_scale_list = _scaler_scale.tolist() if _scaler_scale is not None else []

        return json.dumps({
            'raw_vector': vector.flatten().tolist(),
            'feature_names': list(_feature_names),
            'vector': scaled.flatten().tolist(),
            'scaler_mean': scaler_mean_list,
            'scaler_scale': scaler_scale_list,
            'fusion_artifact': float(artifact),
            'fusion_reality': float(reality),
            'fusion_stress': float(stress_proxy),
            'fusion_probability': float(fusion['final_probability']),
            'fusion_prediction': fusion['prediction'],
            'fusion_confidence': fusion['confidence'],
            'model_prediction': 'FAKE' if model_label == 1 else 'REAL',
            'model_probability': float(model_prob),
            'final_probability': float(final_prob),
            'model_used': model_used,
            'onnx_models': _onnx_models,
            'ensemble_details': ensemble_details,
            'input_quality_score': float(input_quality['score']),
            'input_quality_label': input_quality['label'],
            'quality_flags': input_quality['flags'],
            'customer_verdict': customer_verdict,
            'verdict_headline': verdict_headline,
            'reason_points': reason_points,
            'reason_summary': reason_summary,
            'scan_mode': resolved_scan_mode,
            'scan_profile': scan_profile,
            'explanations': explanations[:6],
            'feature_dim': int(len(scaled.flatten())),
            'metadata': metadata,
            'deep_features_valid': deep_features_valid,
            'deep_features_count': deep_features_count,
            'face_features_count': face_features_count,
            'temporal_features_count': temporal_features_count,
            'debug_info': {
                'n_features_expected': _n_features,
                'n_features_extracted': len(features),
                'n_missing': len(missing),
                'missing': missing[:10],
                'scaled_mean': float(np.mean(scaled)),
                'scaled_std': float(np.std(scaled)),
                'onnx_models_loaded': len(_onnx_models),
            },
        })

    except Exception as exc:
        import traceback
        return json.dumps({
            'error': str(exc),
            'traceback': traceback.format_exc(),
        })


def analyze_video(video_path: str) -> str:
    return json.dumps({'error': 'Hãy dùng extract_features() thay cho analyze_video()'})
