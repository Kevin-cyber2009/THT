# main.py - Deepfake Detector Mobile App (Offline)
"""
Android app for deepfake detection
Runs completely offline with bundled model
- Supports local file selection + URL paste
- Auto-deletes temp downloaded files on exit
- No emoji (uses text symbols safe for all fonts)
"""

import os
import shutil
import tempfile
import threading
import urllib.request
import urllib.parse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(tempfile.gettempdir(), 'deepfake_detector_tmp')
os.makedirs(TEMP_DIR, exist_ok=True)

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.progressbar import ProgressBar
from kivy.uix.scrollview import ScrollView
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.popup import Popup
from kivy.uix.widget import Widget
from kivy.clock import Clock
from kivy.graphics import Color, Rectangle, RoundedRectangle, Ellipse
from kivy.metrics import dp, sp
from kivy.core.window import Window
from kivy.core.clipboard import Clipboard
from kivy.utils import platform

from src.utils import load_config
from src.classifier import VideoClassifier
from src.features import FeatureExtractor
from src.fusion import ScoreFusion

if platform != 'android':
    Window.size = (390, 780)

# ─── Color Palette ─────────────────────────────────────────────────────────────
C_BG      = (0.07, 0.08, 0.10, 1)
C_SURFACE = (0.11, 0.13, 0.17, 1)
C_CARD    = (0.14, 0.16, 0.21, 1)
C_BORDER  = (0.20, 0.23, 0.30, 1)
C_ACCENT  = (0.24, 0.58, 1.00, 1)
C_ACCENT2 = (0.38, 0.82, 0.72, 1)
C_DANGER  = (1.00, 0.33, 0.35, 1)
C_SUCCESS = (0.25, 0.84, 0.56, 1)
C_WARNING = (1.00, 0.72, 0.22, 1)
C_TEXT    = (0.92, 0.93, 0.95, 1)
C_SUBTEXT = (0.52, 0.56, 0.64, 1)
C_MUTED   = (0.30, 0.33, 0.40, 1)


# ─── Helper ────────────────────────────────────────────────────────────────────
def make_label(text, font_size=13, color=None, bold=False,
               halign='left', height=None, **kwargs):
    lbl = Label(
        text=text,
        font_size=sp(font_size),
        color=color or C_TEXT,
        bold=bold,
        halign=halign,
        valign='middle',
        size_hint_y=None,
        height=dp(height or (font_size * 2)),
        **kwargs
    )
    lbl.bind(size=lbl.setter('text_size'))
    return lbl


# ─── Widgets ───────────────────────────────────────────────────────────────────

class StyledButton(Button):
    def __init__(self, bg_color=None, text_color=None, radius=12, **kwargs):
        super().__init__(**kwargs)
        self._c_bg   = list(bg_color or C_ACCENT)
        self._c_text = list(text_color or C_TEXT)
        self._radius = dp(radius)
        self.background_color  = (0, 0, 0, 0)
        self.background_normal = ''
        self.color     = self._c_text
        self.bold      = True
        self.font_size = sp(14)
        self._draw()
        self.bind(pos=self._redraw, size=self._redraw, state=self._on_state)

    def _draw(self, alpha=1.0):
        self.canvas.before.clear()
        with self.canvas.before:
            r, g, b, a = self._c_bg
            Color(r * alpha, g * alpha, b * alpha, a)
            self._rect = RoundedRectangle(
                pos=self.pos, size=self.size, radius=[self._radius])

    def _redraw(self, *a):
        if hasattr(self, '_rect'):
            self._rect.pos  = self.pos
            self._rect.size = self.size

    def _on_state(self, inst, val):
        self._draw(0.72 if val == 'down' else 1.0)


class Divider(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint_y = None
        self.height = dp(1)
        with self.canvas:
            Color(*C_BORDER)
            self._r = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=lambda *a: setattr(self._r, 'pos', self.pos),
                  size=lambda *a: setattr(self._r, 'size', self.size))


class CardBox(BoxLayout):
    def __init__(self, pad=16, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.padding     = [dp(pad)] * 4
        self.spacing     = dp(6)
        self.size_hint_y = None
        with self.canvas.before:
            Color(*C_CARD)
            self._bg = RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(14)])
        self.bind(pos=self._u, size=self._u,
                  minimum_height=self.setter('height'))

    def _u(self, *a):
        self._bg.pos  = self.pos
        self._bg.size = self.size


class ScoreBar(BoxLayout):
    def __init__(self, label, value, color, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.size_hint_y = None
        self.height      = dp(48)
        self.spacing     = dp(4)

        top = BoxLayout(size_hint_y=None, height=dp(18))
        top.add_widget(make_label(label, 12, C_SUBTEXT, height=18))
        top.add_widget(make_label(f'{value*100:.1f}%', 12, list(color),
                                  bold=True, halign='right', height=18))
        self.add_widget(top)

        track = FloatLayout(size_hint_y=None, height=dp(8))
        with track.canvas.before:
            Color(*C_BORDER)
            self._tbg = RoundedRectangle(pos=track.pos, size=track.size, radius=[dp(4)])
        with track.canvas:
            Color(*color)
            self._fill = RoundedRectangle(
                pos=track.pos,
                size=(max(dp(4), track.width * value), track.height),
                radius=[dp(4)]
            )

        def _upd(inst, _):
            self._tbg.pos   = inst.pos
            self._tbg.size  = inst.size
            self._fill.pos  = inst.pos
            self._fill.size = (max(dp(4), inst.width * value), inst.height)

        track.bind(pos=_upd, size=_upd)
        self.add_widget(track)


class MetricRow(BoxLayout):
    def __init__(self, lbl_text, val_text, val_color=None, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'horizontal'
        self.size_hint_y = None
        self.height      = dp(30)
        self.add_widget(make_label(lbl_text, 13, C_SUBTEXT, height=30))
        self.add_widget(make_label(val_text, 13, val_color or C_TEXT,
                                   bold=True, halign='right', height=30))


class ResultCard(CardBox):
    def __init__(self, result, **kwargs):
        super().__init__(pad=20, **kwargs)

        prediction     = result.get('prediction', 'UNKNOWN')
        prob_fake      = result.get('probability_fake', 0)
        prob_real      = 1 - prob_fake
        confidence     = result.get('confidence', 'UNKNOWN')
        artifact_score = result.get('artifact_score', 0)
        reality_score  = result.get('reality_score', 0)
        explanations   = result.get('explanations', [])

        is_fake      = prediction == 'FAKE'
        vc           = list(C_DANGER if is_fake else C_SUCCESS)
        verdict_text = 'DEEPFAKE DETECTED' if is_fake else 'AUTHENTIC VIDEO'
        icon_text    = '!! FAKE !!' if is_fake else '-- REAL --'

        # ── Banner ────────────────────────────────────────────────────────
        banner = BoxLayout(orientation='vertical', size_hint_y=None,
                           height=dp(94), padding=[dp(16), dp(10)], spacing=dp(2))
        with banner.canvas.before:
            Color(vc[0]*0.15, vc[1]*0.15, vc[2]*0.15, 1)
            banner._bbg = RoundedRectangle(pos=banner.pos, size=banner.size,
                                           radius=[dp(12)])
        with banner.canvas:
            Color(*vc)
            banner._bl = Rectangle(pos=banner.pos, size=(dp(4), banner.height))

        def _ub(*a):
            banner._bbg.pos  = banner.pos
            banner._bbg.size = banner.size
            banner._bl.pos   = banner.pos
            banner._bl.size  = (dp(4), banner.height)

        banner.bind(pos=_ub, size=_ub)
        banner.add_widget(make_label(icon_text,    22, vc,       bold=True, halign='center', height=36))
        banner.add_widget(make_label(verdict_text, 15, vc,       bold=True, halign='center', height=24))
        banner.add_widget(make_label(f'Confidence: {confidence}', 11, C_SUBTEXT, halign='center', height=18))
        self.add_widget(banner)
        self.add_widget(Widget(size_hint_y=None, height=dp(10)))

        # ── Probability bars ──────────────────────────────────────────────
        self.add_widget(make_label('PROBABILITY BREAKDOWN', 10, C_MUTED, bold=True, height=18))
        self.add_widget(ScoreBar('Fake', prob_fake, C_DANGER))
        self.add_widget(Widget(size_hint_y=None, height=dp(2)))
        self.add_widget(ScoreBar('Real', prob_real, C_SUCCESS))
        self.add_widget(Widget(size_hint_y=None, height=dp(10)))
        self.add_widget(Divider())
        self.add_widget(Widget(size_hint_y=None, height=dp(10)))

        # ── Scores ────────────────────────────────────────────────────────
        self.add_widget(make_label('ANALYSIS SCORES', 10, C_MUTED, bold=True, height=18))
        self.add_widget(MetricRow('Artifact Score', f'{artifact_score:.3f}', C_WARNING))
        self.add_widget(MetricRow('Reality Score',  f'{reality_score:.3f}',  C_ACCENT2))
        self.add_widget(Widget(size_hint_y=None, height=dp(10)))
        self.add_widget(Divider())
        self.add_widget(Widget(size_hint_y=None, height=dp(10)))

        # ── Key Findings ──────────────────────────────────────────────────
        self.add_widget(make_label('KEY FINDINGS', 10, C_MUTED, bold=True, height=18))
        for exp in (explanations or [])[:5]:
            row = BoxLayout(size_hint_y=None, height=dp(26), spacing=dp(6))
            row.add_widget(make_label('>', 13, C_ACCENT,
                                      size_hint_x=None, width=dp(14), height=26))
            row.add_widget(make_label(str(exp), 12, C_TEXT, height=26))
            self.add_widget(row)

        self.height = self.minimum_height


# ─── Main App ──────────────────────────────────────────────────────────────────

class DeepfakeDetectorApp(App):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.classifier        = None
        self.feature_extractor = None
        self.fusion_engine     = None
        self.selected_video    = None
        self.processing        = False
        self._temp_files       = []

    # ── Build ──────────────────────────────────────────────────────────────────
    def build(self):
        root = BoxLayout(orientation='vertical')
        self._root_ref = root

        with root.canvas.before:
            Color(*C_BG)
            self._bg = Rectangle(pos=root.pos, size=root.size)
        root.bind(pos=self._upd_bg, size=self._upd_bg)

        # Top bar
        topbar = BoxLayout(orientation='horizontal', size_hint_y=None,
                           height=dp(60), padding=[dp(18), dp(10)])
        with topbar.canvas.before:
            Color(*C_SURFACE)
            self._tb_bg = Rectangle(pos=topbar.pos, size=topbar.size)
        topbar.bind(pos=lambda *a: setattr(self._tb_bg, 'pos', topbar.pos),
                    size=lambda *a: setattr(self._tb_bg, 'size', topbar.size))

        title_box = BoxLayout(spacing=dp(8), size_hint_x=None, width=dp(230))
        dot = Widget(size_hint_x=None, width=dp(10))
        with dot.canvas:
            Color(*C_ACCENT)
            Ellipse(pos=(0, dp(6)), size=(dp(10), dp(10)))
        title_box.add_widget(dot)
        title_box.add_widget(make_label('DeepfakeDetector', 17, C_TEXT,
                                        bold=True, halign='left', height=40))
        topbar.add_widget(title_box)

        self.badge_lbl = make_label('* LOADING', 11, list(C_WARNING),
                                    bold=True, halign='right', height=40)
        topbar.add_widget(self.badge_lbl)
        root.add_widget(topbar)

        # Accent line under topbar
        acc = Widget(size_hint_y=None, height=dp(2))
        with acc.canvas:
            Color(*C_ACCENT)
            self._acc_r = Rectangle(pos=acc.pos, size=(dp(110), dp(2)))
        acc.bind(pos=lambda *a: setattr(self._acc_r, 'pos', acc.pos))
        root.add_widget(acc)

        # Body
        body = BoxLayout(orientation='vertical',
                         padding=[dp(16), dp(12)], spacing=dp(10))
        root.add_widget(body)

        self.status_lbl = make_label('Initializing AI model...', 12,
                                     C_SUBTEXT, halign='center', height=22)
        body.add_widget(self.status_lbl)

        self.prog_bar = ProgressBar(size_hint_y=None, height=dp(4),
                                    max=100, value=0)
        body.add_widget(self.prog_bar)

        self.content = BoxLayout(orientation='vertical', spacing=dp(10))
        body.add_widget(self.content)

        self.scroll = ScrollView(size_hint=(1, 1), do_scroll_x=False)
        self.scroll_content = BoxLayout(orientation='vertical', spacing=dp(10),
                                        size_hint_y=None, padding=[0, dp(4)])
        self.scroll_content.bind(minimum_height=self.scroll_content.setter('height'))
        self.scroll.add_widget(self.scroll_content)

        threading.Thread(target=self.load_models, daemon=True).start()
        return root

    def _upd_bg(self, *a):
        self._bg.pos  = self._root_ref.pos
        self._bg.size = self._root_ref.size

    # ── Load models ────────────────────────────────────────────────────────────
    def load_models(self):
        try:
            Clock.schedule_once(lambda dt: setattr(
                self.status_lbl, 'text', 'Loading AI model...'))
            config = load_config(os.path.join(BASE_DIR, 'config.yaml'))
            config.setdefault('preprocessing', {}).update({
                'max_frames': 30, 'target_fps': 3,
                'resize_width': 320, 'resize_height': 180
            })
            self.classifier        = VideoClassifier(config)
            self.classifier.load(os.path.join(BASE_DIR, 'models', 'alpha.pkl'))
            self.feature_extractor = FeatureExtractor(config)
            self.fusion_engine     = ScoreFusion(config)
            Clock.schedule_once(lambda dt: self.show_main_ui())
        except Exception as e:
            # Save to local var before lambda — avoids NameError after except block
            err_msg = f'Error loading model: {e}'
            Clock.schedule_once(lambda dt: (
                setattr(self.status_lbl, 'text', err_msg),
                setattr(self.badge_lbl, 'text', '!! ERROR'),
                setattr(self.badge_lbl, 'color', list(C_DANGER))
            ))

    # ── Main UI ────────────────────────────────────────────────────────────────
    def show_main_ui(self):
        self.status_lbl.text = 'Model ready — select a file or paste a URL.'
        self.badge_lbl.text  = '* READY'
        self.badge_lbl.color = list(C_SUCCESS)
        self.prog_bar.value  = 0
        self.content.clear_widgets()

        # [Select File]  [Paste URL]
        btn_row = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(10))
        sel_btn = StyledButton(text='Select File', size_hint_x=0.5,
                               bg_color=list(C_SURFACE), text_color=list(C_ACCENT))
        url_btn = StyledButton(text='Paste URL', size_hint_x=0.5,
                               bg_color=list(C_SURFACE), text_color=list(C_ACCENT2))
        sel_btn.bind(on_press=self.select_video)
        url_btn.bind(on_press=self.show_url_dialog)
        btn_row.add_widget(sel_btn)
        btn_row.add_widget(url_btn)
        self.content.add_widget(btn_row)

        # File info card
        file_card = CardBox(pad=12)
        file_card.height = dp(50)
        info_row = BoxLayout(size_hint_y=None, height=dp(26), spacing=dp(8))
        self._file_icon_lbl = make_label('[--]', 12, C_MUTED,
                                         size_hint_x=None, width=dp(40), height=26)
        self.file_name_lbl  = make_label('No video selected', 12, C_SUBTEXT, height=26)
        info_row.add_widget(self._file_icon_lbl)
        info_row.add_widget(self.file_name_lbl)
        file_card.add_widget(info_row)
        self.content.add_widget(file_card)

        # Analyze button
        self.analyze_btn = StyledButton(
            text='RUN ANALYSIS',
            size_hint_y=None, height=dp(50),
            bg_color=list(C_ACCENT), text_color=(1, 1, 1, 1)
        )
        self.analyze_btn.disabled = True
        self.analyze_btn.bind(on_press=self.analyze_video)
        self.content.add_widget(self.analyze_btn)
        self.content.add_widget(self.scroll)

    # ── Select local file ──────────────────────────────────────────────────────
    def select_video(self, instance):
        pc = BoxLayout(orientation='vertical', spacing=dp(8),
                       padding=[dp(10), dp(8)])
        with pc.canvas.before:
            Color(*C_BG)
            _r = Rectangle(pos=pc.pos, size=pc.size)
        pc.bind(pos=lambda *a: setattr(_r, 'pos', pc.pos),
                size=lambda *a: setattr(_r, 'size', pc.size))

        fc = FileChooserListView(
            filters=['*.mp4', '*.avi', '*.mov', '*.mkv'], size_hint_y=1)
        pc.add_widget(fc)

        popup = Popup(title='Select Video', content=pc,
                      size_hint=(0.93, 0.88),
                      background_color=list(C_SURFACE),
                      title_color=list(C_TEXT))

        def do_open(b):
            if fc.selection:
                self._set_video(fc.selection[0], is_temp=False)
            popup.dismiss()

        btns = BoxLayout(size_hint_y=None, height=dp(46), spacing=dp(10))
        btns.add_widget(StyledButton(text='Cancel', bg_color=list(C_SURFACE),
                                     text_color=list(C_SUBTEXT),
                                     on_press=lambda b: popup.dismiss()))
        btns.add_widget(StyledButton(text='Open', bg_color=list(C_ACCENT),
                                     text_color=(1, 1, 1, 1), on_press=do_open))
        pc.add_widget(btns)
        popup.open()

    # ── Paste URL dialog ───────────────────────────────────────────────────────
    def show_url_dialog(self, instance):
        pc = BoxLayout(orientation='vertical', spacing=dp(12),
                       padding=[dp(16), dp(14)])
        with pc.canvas.before:
            Color(*C_SURFACE)
            _r = Rectangle(pos=pc.pos, size=pc.size)
        pc.bind(pos=lambda *a: setattr(_r, 'pos', pc.pos),
                size=lambda *a: setattr(_r, 'size', pc.size))

        pc.add_widget(make_label('Enter or paste video URL:', 13, C_TEXT, height=24))

        url_input = TextInput(
            hint_text='https://example.com/video.mp4',
            size_hint_y=None, height=dp(46),
            background_color=list(C_CARD),
            foreground_color=list(C_TEXT),
            cursor_color=list(C_ACCENT),
            font_size=sp(13),
            multiline=False,
            padding=[dp(10), dp(12)]
        )
        # Auto-fill clipboard if it looks like a URL
        try:
            cb = Clipboard.paste()
            if cb and cb.strip().startswith('http'):
                url_input.text = cb.strip()
        except Exception:
            pass

        pc.add_widget(url_input)

        self._url_status = make_label('', 12, C_SUBTEXT, halign='center', height=22)
        pc.add_widget(self._url_status)

        d_btn = StyledButton(text='Download & Analyze',
                             bg_color=list(C_ACCENT2),
                             text_color=(0.05, 0.05, 0.08, 1))

        popup = Popup(title='Paste Video URL', content=pc,
                      size_hint=(0.93, 0.52),
                      background_color=list(C_SURFACE),
                      title_color=list(C_TEXT))

        def do_download(b):
            url = url_input.text.strip()
            if not url.startswith('http'):
                self._url_status.text  = 'Invalid URL — must start with http(s).'
                self._url_status.color = list(C_DANGER)
                return
            d_btn.disabled = True
            self._url_status.text  = 'Downloading...'
            self._url_status.color = list(C_WARNING)

            def _dl():
                try:
                    # Phát hiện link mạng xã hội (YouTube, TikTok, FB...)
                    social_domains = ('youtube.com', 'youtu.be', 'tiktok.com',
                                      'facebook.com', 'fb.watch', 'instagram.com',
                                      'twitter.com', 'x.com', 'vimeo.com')
                    is_social = any(d in url for d in social_domains)

                    dest = os.path.join(TEMP_DIR, 'video_tmp.mp4')
                    # Xóa file cũ nếu còn
                    if os.path.exists(dest):
                        os.remove(dest)

                    if is_social:
                        try:
                            import yt_dlp
                        except ImportError:
                            raise ValueError('Chua cai yt-dlp. Chay: pip install yt-dlp')

                        Clock.schedule_once(lambda dt: setattr(
                            self._url_status, 'text', 'Dang tai tu mang xa hoi...'))

                        tmpl = os.path.join(TEMP_DIR, 'video_tmp.%(ext)s')
                        ydl_opts = {
                            'format': 'best[ext=mp4][height<=480]/best[height<=480]/best',
                            'outtmpl': tmpl,
                            'quiet': True,
                            'no_warnings': True,
                            'overwrites': True,
                        }
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            info = ydl.extract_info(url, download=True)
                            ext  = info.get('ext', 'mp4')

                        dest = os.path.join(TEMP_DIR, f'video_tmp.{ext}')
                        if not os.path.exists(dest):
                            candidates = [
                                os.path.join(TEMP_DIR, f)
                                for f in os.listdir(TEMP_DIR)
                                if f.startswith('video_tmp.')
                            ]
                            if not candidates:
                                raise ValueError('yt-dlp khong luu duoc file video.')
                            dest = max(candidates, key=os.path.getmtime)


                    else:
                        # Direct link — dùng urllib thường
                        Clock.schedule_once(lambda dt: setattr(
                            self._url_status, 'text', 'Dang tai file...'))
                        req = urllib.request.Request(url, headers={
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                                          'AppleWebKit/537.36 Chrome/120.0 Safari/537.36'
                        })
                        with urllib.request.urlopen(req, timeout=30) as resp:
                            content_type = resp.headers.get('Content-Type', '')
                            if 'text/html' in content_type:
                                raise ValueError(
                                    'URL tra ve trang web, khong phai file video.\n'
                                    'Can dung direct link (.mp4 / .avi / .mov)'
                                )
                            data = resp.read()

                        if len(data) < 10_000:
                            raise ValueError(
                                f'File qua nho ({len(data)} bytes) — URL co the bi chan.')

                        with open(dest, 'wb') as f:
                            f.write(data)

                        # Kiem tra magic bytes
                        with open(dest, 'rb') as f:
                            header = f.read(12)
                        valid_boxes = (b'ftyp', b'moov', b'mdat', b'free',
                                       b'wide', b'pnot', b'skip')
                        if len(header) >= 8 and header[4:8] not in valid_boxes:
                            os.remove(dest)
                            raise ValueError(
                                'File khong phai video hop le.\nKiem tra lai URL.')

                    if not os.path.exists(dest) or os.path.getsize(dest) < 10_000:
                        raise ValueError('Tai that bai hoac file rong. Thu lai.')

                    self._temp_files.append(dest)
                    Clock.schedule_once(lambda dt: (
                        self._set_video(dest, is_temp=True),
                        popup.dismiss()
                    ))

                except Exception as e:
                    dl_err = str(e)
                    Clock.schedule_once(lambda dt: (
                        setattr(self._url_status, 'text', dl_err),
                        setattr(self._url_status, 'color', list(C_DANGER)),
                        setattr(d_btn, 'disabled', False)
                    ))

            threading.Thread(target=_dl, daemon=True).start()

        d_btn.bind(on_press=do_download)

        btns = BoxLayout(size_hint_y=None, height=dp(46), spacing=dp(10))
        btns.add_widget(StyledButton(text='Cancel', bg_color=list(C_SURFACE),
                                     text_color=list(C_SUBTEXT),
                                     on_press=lambda b: popup.dismiss()))
        btns.add_widget(d_btn)
        pc.add_widget(btns)
        popup.open()

    # ── Set selected video ─────────────────────────────────────────────────────
    def _set_video(self, path, is_temp=False):
        self.selected_video       = path
        name                      = os.path.basename(path)
        self._file_icon_lbl.text  = '[TMP]' if is_temp else '[FILE]'
        self._file_icon_lbl.color = list(C_WARNING) if is_temp else list(C_ACCENT)
        self.file_name_lbl.text   = name
        self.file_name_lbl.color  = list(C_TEXT)
        self.analyze_btn.disabled = False
        self.badge_lbl.text       = '* LOADED'
        self.badge_lbl.color      = list(C_ACCENT)

    # ── Analyze ────────────────────────────────────────────────────────────────
    def analyze_video(self, instance):
        if not self.selected_video or self.processing:
            return
        self.processing = True
        self.analyze_btn.disabled = True
        self.badge_lbl.text  = '>> ANALYZING'
        self.badge_lbl.color = list(C_WARNING)
        self.prog_bar.value  = 0
        self.scroll_content.clear_widgets()
        threading.Thread(target=self._analyze_thread, daemon=True).start()

    def _analyze_thread(self):
        try:
            Clock.schedule_once(lambda dt: self._upd(10, 'Extracting features...'))
            features_dict, metadata = self.feature_extractor.extract_from_video(
                self.selected_video)

            Clock.schedule_once(lambda dt: self._upd(60, 'Running classifier...'))
            feature_names  = (self.classifier.feature_names
                              or self.feature_extractor.get_feature_names())
            feature_vector = self.feature_extractor.features_to_vector(
                features_dict, feature_names)
            pred, prob = self.classifier.predict(feature_vector.reshape(1, -1))

            Clock.schedule_once(lambda dt: self._upd(80, 'Fusing scores...'))
            artifact_score = self.fusion_engine.compute_artifact_score(features_dict)
            reality_score  = self.fusion_engine.compute_reality_score(features_dict)
            fusion_result  = self.fusion_engine.fuse_scores(
                artifact_score, reality_score, 0.5)
            explanations   = self.fusion_engine.generate_explanation(
                features_dict, fusion_result)

            result = {
                'prediction':       'FAKE' if pred[0] == 1 else 'REAL',
                'probability_fake':  float(prob[0]),
                'probability_real':  float(1 - prob[0]),
                'confidence':        fusion_result['confidence'],
                'artifact_score':    float(artifact_score),
                'reality_score':     float(reality_score),
                'explanations':      explanations,
                'metadata':          metadata
            }
            Clock.schedule_once(lambda dt: self._show_result(result))

        except Exception as e:
            # Save to local var before lambda — avoids NameError after except block
            err_msg = f'Analysis failed: {e}'
            Clock.schedule_once(lambda dt: self._show_error(err_msg))

    def _upd(self, val, text):
        self.prog_bar.value  = val
        self.status_lbl.text = text

    def _show_result(self, result):
        self.processing = False
        self.analyze_btn.disabled = False
        self.prog_bar.value = 100
        verdict = result['prediction']
        self.badge_lbl.text  = f'{"!!" if verdict == "FAKE" else "OK"} {verdict}'
        self.badge_lbl.color = list(C_DANGER if verdict == 'FAKE' else C_SUCCESS)
        self.status_lbl.text = 'Analysis complete.'
        self.scroll_content.clear_widgets()
        self.scroll_content.add_widget(ResultCard(result))
        self.scroll.scroll_y = 1

    def _show_error(self, msg):
        self.processing = False
        self.analyze_btn.disabled = False
        self.badge_lbl.text  = '!! ERROR'
        self.badge_lbl.color = list(C_DANGER)
        self.status_lbl.text = msg
        self.prog_bar.value  = 0

    # ── Cleanup temp files on exit ─────────────────────────────────────────────
    def on_stop(self):
        """Kivy calls this automatically when app closes"""
        for f in self._temp_files:
            try:
                if os.path.isfile(f):
                    os.remove(f)
            except Exception:
                pass
        try:
            if os.path.isdir(TEMP_DIR) and not os.listdir(TEMP_DIR):
                shutil.rmtree(TEMP_DIR, ignore_errors=True)
        except Exception:
            pass


if __name__ == '__main__':
    DeepfakeDetectorApp().run()