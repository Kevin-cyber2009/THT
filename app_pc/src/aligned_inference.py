import copy
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

from .features import FeatureExtractor
from .fusion import ScoreFusion
from .utils import load_config


_DEEP_NAMES = frozenset([
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
])

_SCAN_PROFILES = {
    'quick': {
        'label': 'Quick scan',
        'fps': 4,
        'max_frames': 72,
        'sample_frames': 6,
    },
    'accurate': {
        'label': 'Accurate scan',
        'fps': 8,
        'max_frames': 180,
        'sample_frames': 14,
    },
}


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

    if frame_count < 12:
        score -= 0.30
        flags.append('Too few sampled frames')
    elif frame_count < 24:
        score -= 0.12
        flags.append('Frame count is limited')

    if duration and duration < 2.0:
        score -= 0.25
        flags.append('Video is very short')
    elif duration and duration < 4.0:
        score -= 0.10
        flags.append('Video duration is short')

    if width and height and width * height < 640 * 360:
        score -= 0.18
        flags.append('Source resolution is low')

    if fps and fps < 12:
        score -= 0.08
        flags.append('Source FPS is low')

    score = _clip01(score)
    label = 'HIGH' if score >= 0.85 else 'MEDIUM' if score >= 0.60 else 'LOW'
    return {'score': score, 'label': label, 'flags': flags[:4]}


def _resolve_scan_mode(scan_mode: Optional[str]) -> str:
    normalized = (scan_mode or 'accurate').strip().lower()
    return normalized if normalized in _SCAN_PROFILES else 'accurate'


def _apply_scan_profile(config: dict, scan_mode: str) -> dict:
    runtime = copy.deepcopy(config)
    profile = _SCAN_PROFILES[_resolve_scan_mode(scan_mode)]
    runtime.setdefault('preprocessing', {}).update({
        'fps': profile['fps'],
        'max_frames': profile['max_frames'],
    })
    runtime.setdefault('deep_learning', {})['sample_frames'] = profile['sample_frames']
    return runtime


def blend_probabilities(model_prob: float, fusion_prob: float, input_quality: float) -> float:
    safe_quality = _clip01(input_quality)
    fusion_weight = max(0.18, min(0.40, 0.18 + (1.0 - safe_quality) * 0.22))
    return _clip01(model_prob * (1.0 - fusion_weight) + fusion_prob * fusion_weight)


def derive_confidence(blended_prob: float, model_prob: float, fusion_prob: float, input_quality: float) -> str:
    certainty = abs(blended_prob - 0.5) * 2.0
    agreement = 1.0 - min(abs(model_prob - fusion_prob), 1.0)
    score = _clip01(0.45 * certainty + 0.35 * agreement + 0.20 * _clip01(input_quality))
    if score >= 0.78:
        return 'HIGH'
    if score >= 0.55:
        return 'MEDIUM'
    return 'LOW'


def _build_reason_points(
    artifact: float,
    reality: float,
    stress: float,
    model_prob: float,
    fusion_prob: float,
    quality: dict,
) -> List[str]:
    points: List[str] = []

    if quality['score'] < 0.48 and quality['flags']:
        points.append('video dau vao chua du ro hoac qua ngan')
    if artifact >= 0.62:
        points.append('tin hieu artifact va texture bat thuong kha ro')
    if stress >= 0.58:
        points.append('chuyen dong giua cac frame kem on dinh')
    if model_prob >= 0.68 and fusion_prob >= 0.58:
        points.append('ca model va fusion deu nghieng ve video AI')
    if reality >= 0.62:
        points.append('chi so reality giu duoc su tu nhien kha tot')
    if model_prob <= 0.35 and fusion_prob <= 0.42:
        points.append('ca model va fusion deu nghieng ve video that')

    if not points and quality['flags']:
        points.append('can can nhac chat luong nguon video truoc khi ket luan')
    if not points:
        points.append('cac tin hieu hien tai chua qua cuc doan')

    return points[:3]


def _determine_customer_verdict(
    blended_prob: float,
    confidence: str,
    quality: dict,
) -> Tuple[str, str]:
    if quality['score'] < 0.48:
        return 'INSUFFICIENT_QUALITY', 'Khong du chat luong de ket luan'
    if 0.42 <= blended_prob <= 0.58 or confidence == 'LOW':
        return 'UNCERTAIN', 'Can kiem tra them'
    if blended_prob >= 0.5:
        return 'LIKELY_FAKE', 'Co kha nang la video AI'
    return 'LIKELY_REAL', 'Co kha nang la video that'


def _build_reason_summary(headline: str, reasons: List[str]) -> str:
    if not reasons:
        return headline
    return f"{headline}, vi {reasons[0]}."


def _neutral_value_for(scaler, idx: int) -> float:
    if hasattr(scaler, 'mean_') and idx < len(scaler.mean_):
        return float(scaler.mean_[idx])
    return 0.0


def _coerce_float(value: Any) -> Optional[float]:
    if isinstance(value, (float, int, np.floating, np.integer)):
        return float(value)
    return None


def _parse_prob_map(prob_map: Dict[Any, Any]) -> Optional[float]:
    for key in (1, 1.0, '1', 'fake', 'FAKE'):
        value = _coerce_float(prob_map.get(key))
        if value is not None:
            return value

    numeric_pairs = []
    for key, value in prob_map.items():
        prob_value = _coerce_float(value)
        if prob_value is None:
            continue
        if isinstance(key, (int, float, np.integer, np.floating)):
            numeric_pairs.append((float(key), prob_value))
            continue
        try:
            numeric_pairs.append((float(key), prob_value))
        except (TypeError, ValueError):
            continue

    numeric_pairs.sort(key=lambda item: item[0])
    if len(numeric_pairs) >= 2:
        return numeric_pairs[1][1]

    values = [_coerce_float(value) for value in prob_map.values()]
    values = [value for value in values if value is not None]
    if len(values) >= 2:
        return values[1]
    if len(values) == 1:
        return values[0]
    return None


def _parse_prob_fake(output: list) -> Tuple[float, int]:
    label_raw = output[0]
    if isinstance(label_raw, np.ndarray):
        label = int(label_raw.flatten()[0])
    elif isinstance(label_raw, (list, tuple)):
        label = int(np.array(label_raw).flatten()[0])
    else:
        label = int(label_raw)

    raw_prob = output[1]
    prob_fake = 0.5
    if isinstance(raw_prob, list) and raw_prob:
        first = raw_prob[0]
        if isinstance(first, dict):
            prob_fake = _parse_prob_map(first) or 0.5
        else:
            flattened = [_coerce_float(item) for item in raw_prob]
            flattened = [item for item in flattened if item is not None]
            if len(flattened) >= 2:
                prob_fake = float(flattened[1])
            elif len(flattened) == 1:
                prob_fake = float(flattened[0])
    elif isinstance(raw_prob, np.ndarray):
        if raw_prob.ndim == 2 and raw_prob.shape[1] >= 2:
            prob_fake = float(raw_prob[0, 1])
        elif raw_prob.ndim == 1 and raw_prob.shape[0] >= 2:
            prob_fake = float(raw_prob[1])
        elif raw_prob.size == 1:
            prob_fake = float(raw_prob.flatten()[0])
    elif isinstance(raw_prob, dict):
        prob_fake = _parse_prob_map(raw_prob) or 0.5
    else:
        scalar = _coerce_float(raw_prob)
        if scalar is not None:
            prob_fake = scalar

    return _clip01(prob_fake), label


def run_aligned_inference(
    video_path: str,
    model_stem: str,
    config_path: str,
    scan_mode: str = 'accurate',
) -> Dict[str, Any]:
    import onnxruntime as rt

    base_config = copy.deepcopy(load_config(config_path))
    config = _apply_scan_profile(base_config, scan_mode)
    resolved_scan_mode = _resolve_scan_mode(scan_mode)
    scan_profile = _SCAN_PROFILES[resolved_scan_mode]
    models_dir = os.path.join(os.path.dirname(config_path), 'models')
    config.setdefault('deep_learning', {})['models_dir'] = models_dir
    config['deep_learning']['prefer_onnx'] = True

    available_deep_assets = {
        'resnet50': os.path.exists(os.path.join(models_dir, 'resnet50_features.onnx')),
        'efficientnet_b0': os.path.exists(os.path.join(models_dir, 'efficientnet_b0_features.onnx')),
    }

    scaler_path = os.path.join(models_dir, f'{model_stem}_scaler.pkl')
    onnx_path = os.path.join(models_dir, f'{model_stem}.onnx')
    scaler_blob = joblib.load(scaler_path)
    scaler = scaler_blob['scaler']
    feature_names = scaler_blob.get('feature_names') or []
    expects_deep = any(name in _DEEP_NAMES for name in feature_names)

    config.setdefault('features', {})['use_deep_features'] = expects_deep
    extractor = FeatureExtractor(config)
    fusion = ScoreFusion(config)

    features, metadata = extractor.extract_from_video(video_path)

    vector = []
    missing = []
    for idx, name in enumerate(feature_names):
        value = features.get(name)
        if value is None:
            missing.append(name)
            value = _neutral_value_for(scaler, idx)
        if np.isnan(value) or np.isinf(value):
            value = _neutral_value_for(scaler, idx)
        vector.append(float(value))

    vector = np.array(vector, dtype=np.float64).reshape(1, -1)
    scaled = scaler.transform(vector).astype(np.float32)

    session = rt.InferenceSession(onnx_path)
    output = session.run(None, {session.get_inputs()[0].name: scaled})
    model_prob_fake, label = _parse_prob_fake(output)

    artifact = fusion.compute_artifact_score(features)
    reality = fusion.compute_reality_score(features)
    stress = fusion.compute_stress_proxy(features)
    fusion_result = fusion.fuse_scores(artifact, reality, stress)
    explanations = fusion.generate_explanation(features, fusion_result)
    input_quality = _assess_input_quality(metadata)
    blended_prob = blend_probabilities(model_prob_fake, fusion_result['final_probability'], input_quality['score'])
    blended_prediction = 'FAKE' if blended_prob >= 0.5 else 'REAL'
    blended_confidence = derive_confidence(
        blended_prob,
        model_prob_fake,
        fusion_result['final_probability'],
        input_quality['score'],
    )
    customer_verdict, verdict_headline = _determine_customer_verdict(
        blended_prob,
        blended_confidence,
        input_quality,
    )
    reason_points = _build_reason_points(
        artifact,
        reality,
        stress,
        model_prob_fake,
        fusion_result['final_probability'],
        input_quality,
    )
    reason_summary = _build_reason_summary(verdict_headline, reason_points)

    if input_quality['flags']:
        explanations.append(f"Input quality {input_quality['label']}: " + '; '.join(input_quality['flags']))
    else:
        explanations.append(f"Input quality {input_quality['label']}: du lieu video du tot de danh gia")

    return {
        'video_path': video_path,
        'prediction': blended_prediction,
        'probability_fake': float(blended_prob),
        'probability_real': float(1.0 - blended_prob),
        'confidence': blended_confidence,
        'model_probability_fake': float(model_prob_fake),
        'model_prediction': 'FAKE' if label == 1 else 'REAL',
        'artifact_score': float(artifact),
        'reality_score': float(reality),
        'stress_score': float(stress),
        'fusion_result': fusion_result,
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
        'metadata': metadata,
        'model_stem': model_stem,
        'debug_info': {
            'missing_features': missing[:10],
            'n_missing': len(missing),
            'scaled_mean': float(np.mean(scaled)),
            'scaled_std': float(np.std(scaled)),
            'expects_deep': expects_deep,
            'available_deep_assets': available_deep_assets,
        },
    }
