#!/usr/bin/env python3
# inference.py
"""
Script inference để test model trên videos mới
"""

import argparse
import logging
from pathlib import Path
import numpy as np

from src.features import FeatureExtractor
from src.classifier import VideoClassifier
from src.fusion import ScoreFusion
from src.stress_lab import StressLab
from src.utils import setup_logging, load_config


def predict_single_video(
    video_path: str,
    classifier: VideoClassifier,
    feature_extractor: FeatureExtractor,
    fusion_engine: ScoreFusion = None,
    stress_lab: StressLab = None,
    run_stress_tests: bool = False
):
    """
    Dự đoán cho một video duy nhất
    
    Args:
        video_path: Đường dẫn video
        classifier: Classifier đã train
        feature_extractor: Feature extractor
        fusion_engine: Score fusion engine (optional)
        stress_lab: Stress lab (optional)
        run_stress_tests: Có chạy stress tests không
        
    Returns:
        Dictionary kết quả prediction
    """
    logger = logging.getLogger('hybrid_detector')
    
    logger.info(f"Phân tích video: {video_path}")
    
    # Extract features
    features_dict, metadata = feature_extractor.extract_from_video(video_path)
    
    # Get feature names
    feature_names = classifier.feature_names
    if feature_names is None:
        feature_names = feature_extractor.get_feature_names()
    
    # Convert to vector
    feature_vector = feature_extractor.features_to_vector(
        features_dict,
        feature_names
    )
    
    # Reshape for prediction
    X = feature_vector.reshape(1, -1)
    
    # Predict với classifier
    pred, prob = classifier.predict(X)
    
    result = {
        'video_path': video_path,
        'prediction': 'FAKE' if pred[0] == 1 else 'REAL',
        'probability_fake': float(prob[0]),
        'probability_real': float(1 - prob[0]),
        'metadata': metadata,
        'features': features_dict
    }
    
    # Nếu có fusion engine, tính các component scores
    if fusion_engine:
        artifact_score = fusion_engine.compute_artifact_score(features_dict)
        reality_score = fusion_engine.compute_reality_score(features_dict)
        
        result['artifact_score'] = float(artifact_score)
        result['reality_score'] = float(reality_score)
        
        # Nếu chạy stress tests
        if run_stress_tests and stress_lab:
            logger.info("Chạy stress tests...")
            # Load frames
            frames, _ = feature_extractor.preprocessor.preprocess(video_path)
            
            # Run stress tests
            stress_results = stress_lab.run_stress_tests(frames, feature_extractor)
            stress_score = fusion_engine.compute_stress_score(stress_results)
            
            result['stress_score'] = float(stress_score)
            result['stress_results'] = stress_results
            
            # Fusion scores
            fusion_result = fusion_engine.fuse_scores(
                artifact_score,
                reality_score,
                stress_score
            )
            
            result['fusion_result'] = fusion_result
            
            # Explanations
            explanations = fusion_engine.generate_explanation(
                features_dict,
                fusion_result
            )
            result['explanations'] = explanations
    
    return result


def print_result(result: dict):
    """
    In kết quả ra console theo format đẹp
    
    Args:
        result: Dictionary kết quả từ predict_single_video
    """
    print("\n" + "=" * 80)
    print(f"VIDEO: {Path(result['video_path']).name}")
    print("=" * 80)
    
    # Prediction
    pred = result['prediction']
    prob_fake = result['probability_fake']
    
    if pred == 'FAKE':
        print(f"🔴 DỰ ĐOÁN: {pred} (confidence: {prob_fake:.1%})")
    else:
        print(f"🟢 DỰ ĐOÁN: {pred} (confidence: {1-prob_fake:.1%})")
    
    # Metadata
    metadata = result.get('metadata', {})
    if metadata:
        print(f"\nThông tin video:")
        print(f"  - Số frames: {metadata.get('num_frames', 'N/A')}")
        print(f"  - Resolution: {metadata.get('width', 'N/A')}x{metadata.get('height', 'N/A')}")
        print(f"  - FPS: {metadata.get('fps', 'N/A'):.1f}")
        print(f"  - Duration: {metadata.get('duration', 0):.1f}s")
    
    # Component scores nếu có
    if 'artifact_score' in result:
        print(f"\nComponent Scores:")
        print(f"  - Artifact Score: {result['artifact_score']:.3f}")
        print(f"  - Reality Score: {result['reality_score']:.3f}")
        
        if 'stress_score' in result:
            print(f"  - Stress Score: {result['stress_score']:.3f}")
    
    # Fusion result nếu có
    if 'fusion_result' in result:
        fusion = result['fusion_result']
        print(f"\nFusion Analysis:")
        print(f"  - Final Probability: {fusion['final_probability']:.3f}")
        print(f"  - Prediction: {fusion['prediction']}")
        print(f"  - Confidence: {fusion['confidence']}")
    
    # Explanations nếu có
    if 'explanations' in result:
        print(f"\nGiải thích:")
        for i, exp in enumerate(result['explanations'], 1):
            print(f"  {i}. {exp}")
    
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Inference Hybrid++ Reality Stress Video AI Detector'
    )
    
    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help='Đường dẫn video cần phân tích'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='models/hybrid_detector.pkl',
        help='Đường dẫn model đã train (default: models/hybrid_detector.pkl)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='File cấu hình (default: config.yaml)'
    )
    
    parser.add_argument(
        '--stress_test',
        action='store_true',
        help='Chạy stress tests (chậm hơn nhưng chính xác hơn)'
    )
    
    parser.add_argument(
        '--output_json',
        type=str,
        help='Lưu kết quả ra file JSON'
    )
    
    args = parser.parse_args()
    
    # Check video exists
    if not Path(args.video).exists():
        print(f"❌ Lỗi: Video không tồn tại: {args.video}")
        return
    
    # Check model exists
    if not Path(args.model).exists():
        print(f"❌ Lỗi: Model không tồn tại: {args.model}")
        print(f"   Vui lòng train model trước bằng: python train.py --data_dir <path>")
        return
    
    # Load config
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(config)
    logger.info("=" * 80)
    logger.info("HYBRID++ REALITY STRESS VIDEO AI DETECTOR - INFERENCE")
    logger.info("=" * 80)
    
    # Load model
    logger.info(f"Loading model từ: {args.model}")
    classifier = VideoClassifier(config)
    classifier.load(args.model)
    
    # Initialize components
    logger.info("Khởi tạo feature extractor...")
    feature_extractor = FeatureExtractor(config)
    
    fusion_engine = ScoreFusion(config)
    stress_lab = StressLab(config) if args.stress_test else None
    
    # Predict
    try:
        result = predict_single_video(
            args.video,
            classifier,
            feature_extractor,
            fusion_engine=fusion_engine,
            stress_lab=stress_lab,
            run_stress_tests=args.stress_test
        )
        
        # Print result
        print_result(result)
        
        # Save to JSON nếu được yêu cầu
        if args.output_json:
            import json
            
            # Convert numpy types to Python types for JSON serialization
            def convert_to_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                return obj
            
            result_serializable = convert_to_serializable(result)
            
            with open(args.output_json, 'w', encoding='utf-8') as f:
                json.dump(result_serializable, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✓ Kết quả đã lưu tại: {args.output_json}")
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi phân tích video: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()