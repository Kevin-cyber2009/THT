# run_demo.py

"""
Script demo chính: Chạy full pipeline phân tích video
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

from src.utils import setup_logging, load_config, check_ffmpeg
from src.features import FeatureExtractor
from src.classifier import VideoClassifier
from src.fusion import ScoreFusion
from src.report import ReportGenerator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hybrid++ Reality Stress Video AI Detector - Demo"
    )
    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help='Đường dẫn đến video cần phân tích'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/detector.pkl',
        help='Đường dẫn model đã train (mặc định: models/detector.pkl)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='Thư mục output (mặc định: output/)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='File cấu hình (mặc định: config.yaml)'
    )
    parser.add_argument(
        '--no-stress',
        action='store_true',
        help='Bỏ qua stress tests (chạy nhanh hơn)'
    )
    parser.add_argument(
        '--no-pdf',
        action='store_true',
        help='Không tạo PDF report'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(config)
    logger.info("="*60)
    logger.info("Hybrid++ Reality Stress Video AI Detector - Demo")
    logger.info("="*60)
    
    # Check FFmpeg
    try:
        check_ffmpeg()
        logger.info("✓ FFmpeg khả dụng")
    except RuntimeError as e:
        logger.error(f"✗ {e}")
        return
    
    # Check video
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"✗ Video không tồn tại: {args.video}")
        return
    
    logger.info(f"Video: {video_path}")
    
    # Check model
    model_path = Path(args.model)
    use_model = model_path.exists()
    
    if use_model:
        logger.info(f"Model: {model_path}")
    else:
        logger.warning(f"Model không tồn tại tại {model_path}, chạy rule-based fusion")
    
    # Initialize components
    logger.info("\n" + "="*60)
    logger.info("Bước 1: Khởi tạo components")
    logger.info("="*60)
    
    extractor = FeatureExtractor(config)
    fusion = ScoreFusion(config)
    report_gen = ReportGenerator(config)
    
    # Extract features
    logger.info("\n" + "="*60)
    logger.info("Bước 2: Trích xuất features")
    logger.info("="*60)
    
    try:
        features, metadata = extractor.extract_from_video(str(video_path))
        logger.info(f"✓ Extracted {len(features)} features")
    except Exception as e:
        logger.error(f"✗ Lỗi khi extract features: {e}")
        return
    
    # Run stress tests nếu cần
    stress_results = None
    if not args.no_stress:
        logger.info("\n" + "="*60)
        logger.info("Bước 3: Chạy stress tests")
        logger.info("="*60)
        
        try:
            from src.stress_lab import StressLab
            from src.preprocessing import VideoPreprocessor
            
            preprocessor = VideoPreprocessor(config)
            frames, _ = preprocessor.preprocess(str(video_path))
            
            stress_lab = StressLab(config)
            stress_results = stress_lab.run_stress_tests(frames, extractor)
            
            logger.info(f"✓ Stress tests hoàn tất - stability: {stress_results['aggregate_stability_score']:.3f}")
        except Exception as e:
            logger.warning(f"! Stress tests lỗi: {e}")
            stress_results = {'aggregate_stability_score': 0.5}
    else:
        logger.info("\n" + "="*60)
        logger.info("Bước 3: Bỏ qua stress tests (--no-stress)")
        logger.info("="*60)
        stress_results = {'aggregate_stability_score': 0.5}
    
    # Compute scores
    logger.info("\n" + "="*60)
    logger.info("Bước 4: Tính toán scores")
    logger.info("="*60)
    
    artifact_score = fusion.compute_artifact_score(features)
    reality_score = fusion.compute_reality_score(features)
    stress_score = fusion.compute_stress_score(stress_results)
    
    logger.info(f"Artifact Score: {artifact_score:.3f}")
    logger.info(f"Reality Score:  {reality_score:.3f}")
    logger.info(f"Stress Score:   {stress_score:.3f}")
    
    # Fuse scores
    result = fusion.fuse_scores(artifact_score, reality_score, stress_score)
    
    logger.info("\n" + "="*60)
    logger.info("KẾT QUẢ CUỐI CÙNG")
    logger.info("="*60)
    logger.info(f"Prediction:    {result['prediction']}")
    logger.info(f"Probability:   {result['final_probability']:.3f}")
    logger.info(f"Confidence:    {result['confidence']}")
    
    # Generate explanations
    explanations = fusion.generate_explanation(features, result)
    
    logger.info("\nGiải thích:")
    for i, exp in enumerate(explanations, 1):
        logger.info(f"{i}. {exp}")
    
    # Prepare output
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    video_name = video_path.stem
    json_path = output_dir / f"result_{video_name}.json"
    pdf_path = output_dir / f"report_{video_name}.pdf"
    
    # Save JSON
    logger.info("\n" + "="*60)
    logger.info("Bước 5: Lưu kết quả")
    logger.info("="*60)
    
    output_data = {
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat(),
        'video_path': str(video_path),
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
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ JSON saved: {json_path}")
    
    # Generate PDF
    if not args.no_pdf:
        try:
            report_gen.generate_pdf(
                output_data=output_data,
                output_path=str(pdf_path)
            )
            logger.info(f"✓ PDF saved: {pdf_path}")
        except Exception as e:
            logger.warning(f"! Không tạo được PDF: {e}")
    
    logger.info("\n" + "="*60)
    logger.info("HOÀN TẤT")
    logger.info("="*60)
    
    print("\n" + "="*60)
    print(f"KẾT QUẢ: {result['prediction']} ({result['confidence']} confidence)")
    print(f"Xác suất video FAKE: {result['final_probability']*100:.1f}%")
    print("="*60)


if __name__ == '__main__':
    main()