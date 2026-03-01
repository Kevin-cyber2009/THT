# main.py - Deepfake Detector Mobile App (Offline)
"""
Android app for deepfake detection
Runs completely offline with bundled model
"""

import os

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.progressbar import ProgressBar
from kivy.uix.scrollview import ScrollView
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from kivy.clock import Clock
from kivy.graphics import Color, Rectangle, RoundedRectangle
from kivy.metrics import dp, sp
from kivy.core.window import Window
from kivy.utils import platform

import threading
import numpy as np
from datetime import datetime
import json

# Import ML modules
from src.utils import load_config
from src.classifier import VideoClassifier
from src.features import FeatureExtractor
from src.fusion import ScoreFusion

# Set window size for testing on desktop
if platform != 'android':
    Window.size = (360, 640)

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 

class ResultCard(BoxLayout):
    """Card to display analysis result"""
    
    def __init__(self, result, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.padding = dp(20)
        self.spacing = dp(15)
        self.size_hint_y = None
        self.height = dp(500)
        
        # Background
        with self.canvas.before:
            Color(1, 1, 1, 1)
            self.rect = RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(15)])
        
        self.bind(pos=self._update_rect, size=self._update_rect)
        
        # Prediction
        prediction = result.get('prediction', 'UNKNOWN')
        prob_fake = result.get('probability_fake', 0) * 100
        
        # Color based on prediction
        if prediction == 'FAKE':
            color = [0.9, 0.2, 0.2, 1]  # Red
            emoji = '🚨'
        else:
            color = [0.2, 0.7, 0.3, 1]  # Green
            emoji = '✅'
        
        # Result header
        header_layout = BoxLayout(orientation='vertical', size_hint_y=None, height=dp(120))
        with header_layout.canvas.before:
            Color(*color)
            header_layout.rect = RoundedRectangle(pos=header_layout.pos, size=header_layout.size, radius=[dp(10)])
        header_layout.bind(pos=self._update_header_rect, size=self._update_header_rect)
        
        emoji_label = Label(
            text=emoji,
            font_size=sp(48),
            size_hint_y=None,
            height=dp(60),
            color=[1, 1, 1, 1]
        )
        
        verdict_label = Label(
            text=prediction,
            font_size=sp(32),
            bold=True,
            size_hint_y=None,
            height=dp(40),
            color=[1, 1, 1, 1]
        )
        
        conf_label = Label(
            text=f"Confidence: {result.get('confidence', 'UNKNOWN')}",
            font_size=sp(16),
            size_hint_y=None,
            height=dp(20),
            color=[1, 1, 1, 0.9]
        )
        
        header_layout.add_widget(emoji_label)
        header_layout.add_widget(verdict_label)
        header_layout.add_widget(conf_label)
        self.add_widget(header_layout)
        
        # Probabilities
        prob_layout = BoxLayout(orientation='vertical', size_hint_y=None, height=dp(100), padding=dp(10))
        
        fake_label = Label(
            text=f"Fake: {prob_fake:.1f}%",
            size_hint_y=None,
            height=dp(30),
            color=[0.2, 0.2, 0.2, 1],
            halign='left'
        )
        fake_label.bind(size=fake_label.setter('text_size'))
        
        fake_bar = ProgressBar(
            max=100,
            value=prob_fake,
            size_hint_y=None,
            height=dp(20)
        )
        
        real_label = Label(
            text=f"Real: {100-prob_fake:.1f}%",
            size_hint_y=None,
            height=dp(30),
            color=[0.2, 0.2, 0.2, 1],
            halign='left'
        )
        real_label.bind(size=real_label.setter('text_size'))
        
        prob_layout.add_widget(fake_label)
        prob_layout.add_widget(fake_bar)
        prob_layout.add_widget(Label(size_hint_y=None, height=dp(5)))
        prob_layout.add_widget(real_label)
        
        self.add_widget(prob_layout)
        
        # Scores
        scores_label = Label(
            text=f"Artifact Score: {result.get('artifact_score', 0):.3f}\nReality Score: {result.get('reality_score', 0):.3f}",
            size_hint_y=None,
            height=dp(60),
            color=[0.2, 0.2, 0.2, 1]
        )
        self.add_widget(scores_label)
        
        # Explanations
        exp_label = Label(
            text="Key Findings:",
            size_hint_y=None,
            height=dp(30),
            color=[0.2, 0.2, 0.2, 1],
            bold=True
        )
        self.add_widget(exp_label)
        
        explanations = result.get('explanations', [])
        exp_text = '\n'.join([f"• {exp}" for exp in explanations[:3]])
        
        exp_detail = Label(
            text=exp_text,
            size_hint_y=None,
            height=dp(120),
            color=[0.3, 0.3, 0.3, 1],
            halign='left',
            valign='top',
            text_size=(self.width - dp(40), None)
        )
        self.add_widget(exp_detail)
    
    def _update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size
    
    def _update_header_rect(self, *args):
        if hasattr(self, 'rect'):
            self.rect.pos = self.pos
            self.rect.size = self.size


class DeepfakeDetectorApp(App):
    """Main Kivy app"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.classifier = None
        self.feature_extractor = None
        self.fusion_engine = None
        self.selected_video = None
        self.processing = False
    
    def build(self):
        """Build UI"""
        # Main layout
        self.root = BoxLayout(orientation='vertical', padding=dp(10), spacing=dp(10))
        
        # Set background color
        with self.root.canvas.before:
            Color(0.97, 0.97, 0.97, 1)
            self.bg_rect = Rectangle(pos=self.root.pos, size=self.root.size)
        self.root.bind(pos=self._update_bg, size=self._update_bg)
        
        # Header
        header = BoxLayout(orientation='vertical', size_hint_y=None, height=dp(100))
        
        title = Label(
            text='🔍 Deepfake Detector',
            font_size=sp(28),
            bold=True,
            size_hint_y=None,
            height=dp(60),
            color=[0.1, 0.4, 0.7, 1]
        )
        
        subtitle = Label(
            text='Offline AI-powered detection',
            font_size=sp(14),
            size_hint_y=None,
            height=dp(30),
            color=[0.4, 0.4, 0.4, 1]
        )
        
        header.add_widget(title)
        header.add_widget(subtitle)
        self.root.add_widget(header)
        
        # Status label
        self.status_label = Label(
            text='Loading AI model...',
            size_hint_y=None,
            height=dp(30),
            color=[0.2, 0.2, 0.2, 1]
        )
        self.root.add_widget(self.status_label)
        
        # Progress bar
        self.progress_bar = ProgressBar(size_hint_y=None, height=dp(10))
        self.root.add_widget(self.progress_bar)
        
        # Content area (will be populated after model loads)
        self.content_area = BoxLayout(orientation='vertical', spacing=dp(10))
        self.root.add_widget(self.content_area)
        
        # Load model in background
        threading.Thread(target=self.load_models, daemon=True).start()
        
        return self.root
    
    def load_models(self):
        """Load ML models"""
        try:
            Clock.schedule_once(lambda dt: setattr(self.status_label, 'text', 'Loading model...'))
            
            config = load_config(os.path.join(BASE_DIR, 'config.yaml'))

            
            # Override for mobile optimization
            config['preprocessing'] = config.get('preprocessing', {})
            config['preprocessing']['max_frames'] = 30
            config['preprocessing']['target_fps'] = 3
            config['preprocessing']['resize_width'] = 320
            config['preprocessing']['resize_height'] = 180
            
            # Load classifier
            self.classifier = VideoClassifier(config)
            self.classifier.load(os.path.join(BASE_DIR, 'models', 'x.pkl'))

            
            # Initialize extractors
            self.feature_extractor = FeatureExtractor(config)
            self.fusion_engine = ScoreFusion(config)
            
            Clock.schedule_once(lambda dt: self.show_main_ui())
            
        except Exception as e:
            error_msg = f'Error loading model: {str(e)}'
            Clock.schedule_once(lambda dt: setattr(self.status_label, 'text', error_msg))
    
    def show_main_ui(self):
        """Show main UI after model loaded"""
        self.status_label.text = '✅ Model loaded! Select a video'
        self.progress_bar.value = 0
        
        # Clear content area
        self.content_area.clear_widgets()
        
        # Select button
        select_btn = Button(
            text='📹 Select Video',
            size_hint_y=None,
            height=dp(60),
            background_color=[0.1, 0.6, 0.9, 1],
            font_size=sp(18),
            bold=True
        )
        select_btn.bind(on_press=self.select_video)
        self.content_area.add_widget(select_btn)
        
        # Selected file label
        self.file_label = Label(
            text='No video selected',
            size_hint_y=None,
            height=dp(40),
            color=[0.4, 0.4, 0.4, 1]
        )
        self.content_area.add_widget(self.file_label)
        
        # Analyze button
        self.analyze_btn = Button(
            text='🔍 Analyze',
            size_hint_y=None,
            height=dp(60),
            background_color=[0.2, 0.7, 0.3, 1],
            font_size=sp(18),
            bold=True,
            disabled=True
        )
        self.analyze_btn.bind(on_press=self.analyze_video)
        self.content_area.add_widget(self.analyze_btn)
        
        # Result scroll view
        self.result_scroll = ScrollView(size_hint=(1, 1))
        self.result_container = BoxLayout(orientation='vertical', size_hint_y=None, spacing=dp(10))
        self.result_container.bind(minimum_height=self.result_container.setter('height'))
        self.result_scroll.add_widget(self.result_container)
        self.content_area.add_widget(self.result_scroll)
    
    def select_video(self, instance):
        """Show file chooser"""
        content = BoxLayout(orientation='vertical')
        
        filechooser = FileChooserIconView(
            filters=['*.mp4', '*.avi', '*.mov', '*.mkv']
        )
        
        button_layout = BoxLayout(size_hint_y=None, height=dp(50), spacing=dp(10))
        
        cancel_btn = Button(text='Cancel')
        select_btn = Button(text='Select')
        
        def cancel(btn):
            popup.dismiss()
        
        def select(btn):
            if filechooser.selection:
                self.selected_video = filechooser.selection[0]
                self.file_label.text = f'✓ {os.path.basename(self.selected_video)}'
                self.analyze_btn.disabled = False
            popup.dismiss()
        
        cancel_btn.bind(on_press=cancel)
        select_btn.bind(on_press=select)
        
        button_layout.add_widget(cancel_btn)
        button_layout.add_widget(select_btn)
        
        content.add_widget(filechooser)
        content.add_widget(button_layout)
        
        popup = Popup(
            title='Select Video',
            content=content,
            size_hint=(0.9, 0.9)
        )
        popup.open()
    
    def analyze_video(self, instance):
        """Analyze selected video"""
        if not self.selected_video or self.processing:
            return
        
        self.processing = True
        self.analyze_btn.disabled = True
        self.status_label.text = 'Analyzing...'
        self.progress_bar.max = 100
        self.progress_bar.value = 0
        
        # Run analysis in background
        threading.Thread(target=self._analyze_thread, daemon=True).start()
    
    def _analyze_thread(self):
        """Analysis thread"""
        try:
            # Update progress
            Clock.schedule_once(lambda dt: self._update_progress(10, 'Extracting features...'))
            
            # Extract features
            features_dict, metadata = self.feature_extractor.extract_from_video(
                self.selected_video,
                max_frames=30,
                fast_mode=True
            )
            
            Clock.schedule_once(lambda dt: self._update_progress(60, 'Running classifier...'))
            
            # Predict
            feature_names = self.classifier.feature_names or self.feature_extractor.get_feature_names()
            feature_vector = self.feature_extractor.features_to_vector(features_dict, feature_names)
            X = feature_vector.reshape(1, -1)
            
            pred, prob = self.classifier.predict(X)
            
            Clock.schedule_once(lambda dt: self._update_progress(80, 'Computing scores...'))
            
            # Fusion
            artifact_score = self.fusion_engine.compute_artifact_score(features_dict)
            reality_score = self.fusion_engine.compute_reality_score(features_dict)
            fusion_result = self.fusion_engine.fuse_scores(artifact_score, reality_score, 0.5)
            explanations = self.fusion_engine.generate_explanation(features_dict, fusion_result)
            
            # Prepare result
            result = {
                'prediction': 'FAKE' if pred[0] == 1 else 'REAL',
                'probability_fake': float(prob[0]),
                'probability_real': float(1 - prob[0]),
                'confidence': fusion_result['confidence'],
                'artifact_score': float(artifact_score),
                'reality_score': float(reality_score),
                'explanations': explanations,
                'metadata': metadata
            }
            
            Clock.schedule_once(lambda dt: self._show_result(result))
            
        except Exception as e:
            error_msg = f'Analysis failed: {str(e)}'
            Clock.schedule_once(lambda dt: self._show_error(error_msg))
    
    def _update_progress(self, value, text):
        """Update progress bar and text"""
        self.progress_bar.value = value
        self.status_label.text = text
    
    def _show_result(self, result):
        """Display analysis result"""
        self.processing = False
        self.analyze_btn.disabled = False
        self.status_label.text = f'✅ Analysis complete! Result: {result["prediction"]}'
        self.progress_bar.value = 100
        
        # Clear previous results
        self.result_container.clear_widgets()
        
        # Add result card
        result_card = ResultCard(result)
        self.result_container.add_widget(result_card)
        
        # Scroll to result
        self.result_scroll.scroll_y = 0
    
    def _show_error(self, error_msg):
        """Show error"""
        self.processing = False
        self.analyze_btn.disabled = False
        self.status_label.text = error_msg
        self.progress_bar.value = 0
    
    def _update_bg(self, *args):
        """Update background rectangle"""
        self.bg_rect.pos = self.root.pos
        self.bg_rect.size = self.root.size


if __name__ == '__main__':
    DeepfakeDetectorApp().run()
