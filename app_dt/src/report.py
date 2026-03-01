# src/report.py

"""
Module report: Tạo báo cáo PDF và JSON từ kết quả phân tích
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import logging

import matplotlib
matplotlib.use('Agg')  # Backend không cần display
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT

from .utils import load_config


logger = logging.getLogger('hybrid_detector.report')


class ReportGenerator:
    """
    Class tạo báo cáo phân tích video
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Khởi tạo ReportGenerator
        
        Args:
            config: Dictionary cấu hình
        """
        if config is None:
            config = load_config()
        
        self.report_config = config.get('report', {})
        self.page_size = A4
        self.plot_dpi = self.report_config.get('plot_dpi', 100)
        self.include_plots = self.report_config.get('include_plots', True)
        
        # Set plot style
        plt_style = self.report_config.get('plot_style', 'seaborn-v0_8-darkgrid')
        try:
            plt.style.use(plt_style)
        except:
            plt.style.use('default')
        
        logger.info("ReportGenerator initialized")
    
    def generate_score_plot(self, scores: Dict[str, float], output_path: str):
        """
        Tạo biểu đồ thanh scores
        
        Args:
            scores: Dictionary scores
            output_path: Đường dẫn lưu plot
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        
        score_names = ['Artifact\nScore', 'Reality\nScore', 'Stress\nScore', 'Final\nProbability']
        score_values = [
            scores['artifact_score'],
            scores['reality_score'],
            scores['stress_score'],
            scores['final_probability']
        ]
        
        colors_map = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        
        bars = ax.barh(score_names, score_values, color=colors_map, alpha=0.7)
        
        # Thêm giá trị trên thanh
        for bar, value in zip(bars, score_values):
            ax.text(value + 0.02, bar.get_y() + bar.get_height()/2, 
                   f'{value:.2f}', va='center', fontsize=10)
        
        ax.set_xlim(0, 1.0)
        ax.set_xlabel('Score', fontsize=12)
        ax.set_title('Analysis Scores', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.plot_dpi, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"Score plot saved: {output_path}")
    
    def generate_feature_importance_plot(
        self, 
        features: Dict[str, float],
        output_path: str,
        top_n: int = 10
    ):
        """
        Tạo biểu đồ top features
        
        Args:
            features: Dictionary features
            output_path: Đường dẫn lưu
            top_n: Số features hiển thị
        """
        # Sort features theo absolute value
        sorted_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
        
        names = [f[0] for f in sorted_features]
        values = [f[1] for f in sorted_features]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        colors_list = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]
        
        ax.barh(names, values, color=colors_list, alpha=0.7)
        ax.set_xlabel('Feature Value', fontsize=12)
        ax.set_title(f'Top {top_n} Features', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.plot_dpi, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"Feature plot saved: {output_path}")
    
    def generate_pdf(
        self,
        output_data: Dict[str, Any],
        output_path: str
    ):
        """
        Tạo PDF report đầy đủ
        
        Args:
            output_data: Dictionary kết quả phân tích
            output_path: Đường dẫn PDF output
        """
        logger.info(f"Generating PDF report: {output_path}")
        
        # Tạo document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=self.page_size,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm
        )
        
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2c3e50'),
            alignment=TA_CENTER,
            spaceAfter=12
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=10
        )
        
        # Title
        story.append(Paragraph("Video AI Detection Report", title_style))
        story.append(Spacer(1, 0.5*cm))
        
        # Metadata table
        metadata = output_data.get('metadata', {})
        meta_data = [
            ['Video:', Path(output_data['video_path']).name],
            ['Timestamp:', output_data['timestamp']],
            ['Frames:', str(metadata.get('num_frames', 'N/A'))],
            ['FPS:', str(metadata.get('fps', 'N/A'))],
            ['Duration:', f"{metadata.get('duration', 0):.1f}s"],
        ]
        
        meta_table = Table(meta_data, colWidths=[4*cm, 12*cm])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        
        story.append(meta_table)
        story.append(Spacer(1, 1*cm))
        
        # Result section
        story.append(Paragraph("Analysis Result", heading_style))
        
        prediction = output_data['prediction']
        confidence = output_data['confidence']
        probability = output_data['final_probability']
        
        # Prediction box
        pred_color = colors.HexColor('#e74c3c') if prediction == 'FAKE' else colors.HexColor('#2ecc71')
        
        result_data = [
            ['PREDICTION:', prediction],
            ['CONFIDENCE:', confidence],
            ['PROBABILITY:', f"{probability:.1%}"],
        ]
        
        result_table = Table(result_data, colWidths=[5*cm, 11*cm])
        result_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#ecf0f1')),
            ('TEXTCOLOR', (1, 0), (1, 0), pred_color),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        
        story.append(result_table)
        story.append(Spacer(1, 1*cm))
        
        # Scores section
        story.append(Paragraph("Component Scores", heading_style))
        
        scores = output_data['scores']
        scores['final_probability'] = probability
        
        # Generate score plot
        if self.include_plots:
            plot_path = Path(output_path).parent / '_temp_scores.png'
            self.generate_score_plot(scores, str(plot_path))
            
            if plot_path.exists():
                img = Image(str(plot_path), width=14*cm, height=9*cm)
                story.append(img)
                story.append(Spacer(1, 0.5*cm))
                plot_path.unlink()  # Delete temp file
        
        # Explanations
        story.append(Paragraph("Detailed Explanation", heading_style))
        
        for i, exp in enumerate(output_data.get('explanations', []), 1):
            bullet_text = f"{i}. {exp}"
            story.append(Paragraph(bullet_text, styles['Normal']))
            story.append(Spacer(1, 0.3*cm))
        
        # Page break
        story.append(PageBreak())
        
        # Feature details
        story.append(Paragraph("Key Features", heading_style))
        
        features = output_data.get('features', {})
        
        if self.include_plots:
            feat_plot_path = Path(output_path).parent / '_temp_features.png'
            self.generate_feature_importance_plot(features, str(feat_plot_path), top_n=10)
            
            if feat_plot_path.exists():
                img = Image(str(feat_plot_path), width=14*cm, height=10*cm)
                story.append(img)
                feat_plot_path.unlink()
        
        story.append(Spacer(1, 0.5*cm))
        
        # Feature table
        feat_data = [['Feature Name', 'Value']]
        for name, value in sorted(features.items())[:15]:  # Top 15
            feat_data.append([name, f"{value:.4f}"])
        
        feat_table = Table(feat_data, colWidths=[10*cm, 6*cm])
        feat_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        
        story.append(feat_table)
        story.append(Spacer(1, 1*cm))
        
        # Footer
        footer_text = f"Generated by Hybrid++ Reality Stress Detector v{output_data.get('version', '1.0.0')}"
        story.append(Paragraph(footer_text, styles['Normal']))
        
        # Build PDF
        doc.build(story)
        
        logger.info(f"PDF report generated: {output_path}")
    
    def save_json_report(
        self,
        output_data: Dict[str, Any],
        output_path: str
    ):
        """
        Lưu báo cáo dạng JSON
        
        Args:
            output_data: Dictionary kết quả
            output_path: Đường dẫn JSON
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"JSON report saved: {output_path}")