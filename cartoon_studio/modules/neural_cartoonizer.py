from __future__ import annotations

from typing import Optional
import cv2
import numpy as np

from cartoon_studio.modules.style_manager import StyleProfile, StyleManager

class NeuralCartoonizer:
    """High-Performance 2D Neural & Cel-Shaded Cartoon Engine with multi-style synthesis."""

    def __init__(self, style_name: str = "Classic 2D Cartoon"):
        self.profile: StyleProfile = StyleManager.get_profile(style_name)

    def set_style(self, style_name: str) -> None:
        """Switch active cartoon style profile."""
        self.profile = StyleManager.get_profile(style_name)

    def cartoonize_frame(
        self,
        frame_bgr: np.ndarray,
        intensity: float = 0.85,
        line_weight: float = 1.0,
    ) -> np.ndarray:
        """Transform a real-life video frame into a consistent 2D cartoon animation frame."""
        intensity = np.clip(intensity, 0.1, 1.0)
        line_weight = np.clip(line_weight, 0.2, 2.5)

        # 1. Surface smoothing & Bilateral filtering (Cel-surface simplification)
        smoothed = self._apply_bilateral_smoothing(frame_bgr)

        # 2. Extract 2D Cartoon Line Art
        lines = self._generate_cartoon_line_art(frame_bgr, line_weight)

        # 3. Color quantization (Flat 2D cartoon cel-shading)
        quantized = self._quantize_colors(smoothed, self.profile.quantization_levels)

        # 4. Color grading, saturation, and contrast adjustment
        graded = self._apply_color_grading(quantized)

        # 5. Style-specific rendering (Anime bloom, Comic halftone/shadow, Hand-drawn pencil)
        stylized = self._apply_style_effects(graded, frame_bgr)

        # 6. Composite cartoon line art onto stylized surface
        # Lines are single channel (0 = black line, 255 = paper/fill)
        lines_3ch = cv2.cvtColor(lines, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
        cartoon_bgr = (stylized.astype(np.float32) * lines_3ch).astype(np.uint8)

        # 7. Apply animation intensity blending with original frame
        if intensity < 0.99:
            output = cv2.addWeighted(cartoon_bgr, intensity, frame_bgr, 1.0 - intensity, 0.0)
        else:
            output = cartoon_bgr

        return output

    def _apply_bilateral_smoothing(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Perform multi-pass bilateral filtering to preserve major edges while smoothing micro-textures."""
        d = self.profile.bilateral_d
        sc = self.profile.bilateral_sigma_color
        ss = self.profile.bilateral_sigma_space

        # Multi-scale pyramid for fast, smooth painterly look
        small = cv2.pyrDown(frame_bgr)
        # Apply 2 passes of bilateral filter for flat animation surfaces
        b1 = cv2.bilateralFilter(small, d, sc, ss)
        b2 = cv2.bilateralFilter(b1, d, sc, ss)
        # Upscale back to original resolution
        smooth = cv2.pyrUp(b2)
        # Ensure dimensions match exactly in case of odd width/height
        if smooth.shape[:2] != frame_bgr.shape[:2]:
            smooth = cv2.resize(smooth, (frame_bgr.shape[1], frame_bgr.shape[0]))
        return smooth

    def _generate_cartoon_line_art(self, frame_bgr: np.ndarray, line_weight: float) -> np.ndarray:
        """Generate clean, expressive 2D animation outlines using adaptive thresholding and DoG."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Apply median blur to reduce photographic sensor noise
        ksize = self.profile.edge_blur_ksize
        if ksize % 2 == 0:
            ksize += 1
        blurred = cv2.medianBlur(gray, ksize)

        # Adaptive thresholding for clean ink lines
        block_size = 9
        c_val = 2.0
        edges = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c_val
        )

        # Additional Canny gradient reinforcement for strong silhouette contours
        canny = cv2.Canny(
            blurred,
            self.profile.edge_threshold_low,
            self.profile.edge_threshold_high
        )
        canny_inv = cv2.bitwise_not(canny)

        # Combine adaptive lines with Canny contours
        combined_lines = cv2.bitwise_and(edges, canny_inv)

        # Line thickness / weight morphological adjustment
        thick = max(1, int(round(self.profile.line_thickness * line_weight)))
        if thick > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thick, thick))
            combined_lines = cv2.erode(combined_lines, kernel, iterations=1)

        return combined_lines

    def _quantize_colors(self, image_bgr: np.ndarray, levels: int) -> np.ndarray:
        """Posterize and quantize colors into discrete cel-shaded palette tones."""
        # Level bin size
        levels = max(2, min(32, levels))
        div = 256 // levels
        quantized = (image_bgr // div) * div + (div // 2)
        return np.clip(quantized, 0, 255).astype(np.uint8)

    def _apply_color_grading(self, image_bgr: np.ndarray) -> np.ndarray:
        """Enhance color saturation, contrast, and apply style tint."""
        # Saturation in HSV space
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= self.profile.saturation_boost
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        # Contrast adjustment on Value channel
        mean_v = np.mean(hsv[:, :, 2])
        hsv[:, :, 2] = mean_v + self.profile.contrast_multiplier * (hsv[:, :, 2] - mean_v)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)

        graded = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

        # Channel tinting
        tb, tg, tr = self.profile.tint_bgr
        graded[:, :, 0] *= tb
        graded[:, :, 1] *= tg
        graded[:, :, 2] *= tr

        return np.clip(graded, 0, 255).astype(np.uint8)

    def _apply_style_effects(self, image_bgr: np.ndarray, orig_bgr: np.ndarray) -> np.ndarray:
        """Apply specialized shader techniques tailored to each of the 5 styles."""
        style_id = self.profile.id
        result = image_bgr.copy()

        # 1. Anime Style: Aesthetic atmospheric bloom and soft highlight glow
        if style_id == "anime":
            if self.profile.bloom_strength > 0.01:
                # Extract highlights
                gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                _, highlight_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
                highlights = cv2.bitwise_and(result, result, mask=highlight_mask)
                bloom = cv2.GaussianBlur(highlights, (35, 35), 0)
                # Screen blend mode: 1 - (1-A)*(1-B)
                a = result.astype(np.float32) / 255.0
                b = (bloom.astype(np.float32) / 255.0) * self.profile.bloom_strength
                screen = 1.0 - (1.0 - a) * (1.0 - b)
                result = (screen * 255.0).astype(np.uint8)

        # 2. Comic-Book Style: Deep black graphic shadows and high-contrast ink
        elif style_id == "comic_book":
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            # Deep shadow pooling (comic inking)
            shadow_mask = gray < 60
            result[shadow_mask] = (result[shadow_mask].astype(np.float32) * 0.35).astype(np.uint8)
            # Crisp highlights
            highlight_mask = gray > 215
            result[highlight_mask] = np.minimum(255, result[highlight_mask].astype(np.float32) * 1.15).astype(np.uint8)

        # 3. Hand-Drawn Style: Soft paper grain and subtle watercolor wash
        elif style_id == "hand_drawn":
            h, w = result.shape[:2]
            # Generate subtle textured paper grain pattern
            grain = np.random.normal(loc=0.0, scale=4.0, size=(h, w)).astype(np.float32)
            grain_3ch = np.repeat(grain[:, :, np.newaxis], 3, axis=2)
            result = np.clip(result.astype(np.float32) + grain_3ch, 0, 255).astype(np.uint8)
            # Soft watercolor edge dispersion
            result = cv2.bilateralFilter(result, 5, 40, 40)

        # 4. Colorful Animated Movie Style: Disney/Pixar 2D storybook specular highlights
        elif style_id == "animated_movie":
            # Warm ambient radiance and smooth gradient preservation
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
            # Boost color warmth (A and B channels)
            lab[:, :, 1] = np.clip(lab[:, :, 1] * 1.08, 0, 255) # slight magenta/pink
            lab[:, :, 2] = np.clip(lab[:, :, 2] * 1.12, 0, 255) # golden warmth
            result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        return result

