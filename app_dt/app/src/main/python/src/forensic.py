# src/forensic.py
"""
Module forensic: Phân tích forensic (FFT/DCT, PRNU residual, optical flow)
Optimized: parallel execution + vectorized numpy
Fix: shape mismatch trong optical flow corrcoef
Logic tính toán 100% giống bản gốc — không ảnh hưởng accuracy
"""

import cv2
import numpy as np
from scipy import fft, signal
from typing import Dict, Any, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from .utils import load_config, safe_divide


logger = logging.getLogger('hybrid_detector.forensic')


class ForensicAnalyzer:

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = load_config()

        self.forensic_config = config.get('forensic', {})
        self.fft_components  = self.forensic_config.get('fft_components', 10)
        self.dct_components  = self.forensic_config.get('dct_components', 10)
        self.spectrum_bins   = self.forensic_config.get('spectrum_bins', 8)

        self.prnu_kernel = self.forensic_config.get('prnu_denoise_kernel', 5)
        self.prnu_method = self.forensic_config.get('prnu_method', 'bilateral')

        self.flow_quality    = self.forensic_config.get('optical_flow_quality', 0.01)
        self.flow_min_dist   = self.forensic_config.get('optical_flow_min_distance', 10)
        self.flow_block_size = self.forensic_config.get('optical_flow_block_size', 7)

        logger.info("ForensicAnalyzer initialized")

    # ─────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────

    @staticmethod
    def _to_gray_float(frames: np.ndarray) -> np.ndarray:
        if frames.ndim == 4 and frames.shape[-1] == 3:
            return np.mean(frames, axis=-1)
        return frames.squeeze(-1)

    @staticmethod
    def _to_gray_uint8(frames: np.ndarray) -> np.ndarray:
        u8 = (frames * 255).astype(np.uint8)
        if u8.ndim == 4 and u8.shape[-1] == 3:
            return np.array([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in u8])
        return u8.squeeze(-1)

    # ─────────────────────────────────────────────
    # FFT — vectorized numpy (kết quả y hệt scipy loop)
    # ─────────────────────────────────────────────

    def compute_fft_features(self, frames: np.ndarray) -> Dict[str, float]:
        gray     = self._to_gray_float(frames)
        features = {}

        f_shift = np.fft.fftshift(np.fft.fft2(gray))
        mag_log = np.log1p(np.abs(f_shift))  # (N, H, W)

        features['fft_mean'] = float(np.mean(mag_log))
        features['fft_std']  = float(np.std(mag_log))
        features['fft_max']  = float(np.max(mag_log))

        h, w = mag_log.shape[1], mag_log.shape[2]
        features['fft_high_freq_energy'] = float(np.mean(mag_log[:, :h//4, :w//4]))

        mean_mag = np.mean(mag_log, axis=0)
        cy, cx   = h // 2, w // 2
        y, x     = np.ogrid[:h, :w]
        r_int    = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
        step     = max(h, w) // (2 * self.spectrum_bins)

        radial_profile = []
        for i in range(self.spectrum_bins):
            mask = (r_int == i * step)
            if np.any(mask):
                radial_profile.append(float(np.mean(mean_mag[mask])))

        if radial_profile:
            features['fft_radial_slope'] = float(
                np.polyfit(range(len(radial_profile)), radial_profile, 1)[0]
            )
        else:
            features['fft_radial_slope'] = 0.0

        logger.debug(f"FFT features: {features}")
        return features

    # ─────────────────────────────────────────────
    # DCT — vectorized scipy (100% giống cũ)
    # ─────────────────────────────────────────────

    def compute_dct_features(self, frames: np.ndarray) -> Dict[str, float]:
        gray      = self._to_gray_float(frames)
        features  = {}

        dct_stack = fft.dctn(gray, axes=(-2, -1), norm='ortho')  # (N, H, W)

        features['dct_mean']    = float(np.mean(dct_stack))
        features['dct_std']     = float(np.std(dct_stack))
        features['dct_dc_mean'] = float(np.mean(dct_stack[:, 0, 0]))

        ac = dct_stack.copy()
        ac[:, 0, 0] = 0
        features['dct_ac_energy'] = float(np.mean(np.abs(ac)))

        logger.debug(f"DCT features: {features}")
        return features

    # ─────────────────────────────────────────────
    # PRNU — 100% giống cũ
    # ─────────────────────────────────────────────

    def compute_prnu_residual(self, frames: np.ndarray) -> Dict[str, float]:
        features = {}
        gray_u8  = self._to_gray_uint8(frames)

        residuals = []
        for frame in gray_u8:
            if self.prnu_method == 'bilateral':
                denoised = cv2.bilateralFilter(frame, self.prnu_kernel, 75, 75)
            else:
                denoised = cv2.GaussianBlur(frame, (self.prnu_kernel, self.prnu_kernel), 0)
            residuals.append(frame.astype(np.float32) - denoised.astype(np.float32))

        residuals = np.array(residuals)

        features['prnu_mean'] = float(np.mean(np.abs(residuals)))
        features['prnu_std']  = float(np.std(residuals))

        if len(residuals) > 0:
            r        = residuals[0]
            autocorr = signal.correlate2d(r, r, mode='same')
            center   = autocorr[autocorr.shape[0]//2, autocorr.shape[1]//2]

            if center != 0:
                autocorr_norm = autocorr / center
                h, w   = autocorr_norm.shape
                offset = autocorr_norm[h//2-5:h//2+5, w//2-5:w//2+5]
                features['prnu_autocorr'] = float(np.mean(np.abs(offset)))
            else:
                features['prnu_autocorr'] = 0.0
        else:
            features['prnu_autocorr'] = 0.0

        if len(residuals) > 1:
            features['prnu_temporal_consistency'] = float(
                np.var([np.mean(r) for r in residuals])
            )
        else:
            features['prnu_temporal_consistency'] = 0.0

        logger.debug(f"PRNU features: {features}")
        return features

    # ─────────────────────────────────────────────
    # Optical Flow — fix shape mismatch
    # ─────────────────────────────────────────────

    def compute_optical_flow(self, frames: np.ndarray) -> Dict[str, float]:
        features = {}
        gray_u8  = self._to_gray_uint8(frames)

        flow_magnitudes = []
        flow_angles     = []

        for i in range(len(gray_u8) - 1):
            flow = cv2.calcOpticalFlowFarneback(
                gray_u8[i], gray_u8[i + 1], None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_magnitudes.append(mag)
            flow_angles.append(ang)

        if not flow_magnitudes:
            return {
                'flow_mean_magnitude':       0.0,
                'flow_std_magnitude':        0.0,
                'flow_smoothness':           0.0,
                'flow_temporal_consistency': 0.0,
            }

        mag_stack = np.array(flow_magnitudes)

        features['flow_mean_magnitude'] = float(np.mean(mag_stack))
        features['flow_std_magnitude']  = float(np.std(mag_stack))

        grad_x = np.gradient(mag_stack, axis=2)
        grad_y = np.gradient(mag_stack, axis=1)
        features['flow_smoothness'] = float(
            1.0 / (1.0 + np.mean(grad_x**2 + grad_y**2))
        )

        if len(mag_stack) > 1:
            corrs = []
            for i in range(len(mag_stack) - 1):
                a       = mag_stack[i].flatten()
                b       = mag_stack[i + 1].flatten()
                # ── FIX: cắt về cùng độ dài để tránh broadcast error ──
                min_len = min(len(a), len(b))
                c = np.corrcoef(a[:min_len], b[:min_len])[0, 1]
                if not np.isnan(c):
                    corrs.append(c)
            features['flow_temporal_consistency'] = float(np.mean(corrs)) if corrs else 0.0
        else:
            features['flow_temporal_consistency'] = 0.0

        logger.debug(f"Optical flow features: {features}")
        return features

    # ─────────────────────────────────────────────
    # analyze — PARALLEL
    # ─────────────────────────────────────────────

    def analyze(self, frames: np.ndarray) -> Dict[str, Any]:
        """
        FFT / DCT / PRNU / Optical Flow chạy SONG SONG.
        Kết quả từng hàm 100% giống cũ — chỉ thứ tự chạy thay đổi.
        """
        logger.info("Bắt đầu forensic analysis (parallel)...")

        tasks = {
            'fft':  self.compute_fft_features,
            'dct':  self.compute_dct_features,
            'prnu': self.compute_prnu_residual,
            'flow': self.compute_optical_flow,
        }

        all_features = {}

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(fn, frames): name
                for name, fn in tasks.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    all_features.update(result)
                    logger.debug(f"✓ {name} done ({len(result)} features)")
                except Exception as e:
                    logger.error(f"✗ {name} failed: {e}")

        logger.info(f"Forensic analysis hoàn tất — {len(all_features)} features")
        return all_features