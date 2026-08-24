import sys
from pathlib import Path

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from wavelet_denoise import (
    imwrite_checked,
    read_rgba_luma,
    save_gray,
    save_with_luma,
    structure_weight,
    white_annotation_mask,
)
from wavelet_multi_parabola import (
    Component,
    apply_noise_correction,
    build_wavelet_residual,
    find_best_component,
    fit_component,
)

from .models import CorrectionLayer, Roi


def parse_floats(text: str, default: list[float]) -> list[float]:
    text = text.strip()
    if not text:
        return default
    return [float(v) for v in text.replace(",", " ").split()]


def parse_sign(text: str) -> float:
    value = float(text)
    return -1.0 if value < 0 else 1.0


def direction_sign(direction: str) -> float:
    direction = direction.lower().strip()
    if direction in ("up", "left"):
        return -1.0
    if direction in ("down", "right"):
        return 1.0
    return -1.0


def is_horizontal_direction(direction: str) -> bool:
    return direction.lower().strip() in ("left", "right")


def roi_weight(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    wy = np.hanning(max(h, 3)).astype(np.float32)
    wx = np.hanning(max(w, 3)).astype(np.float32)
    wy = np.maximum(wy, 0.08)
    wx = np.maximum(wx, 0.08)
    return wy[:, None] * wx[None, :]


def luma_to_rgb(rgb: np.ndarray, luma: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] = np.clip(luma * 255, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


class CorrectionBackend:
    def __init__(self, input_path: Path):
        self.input_path = input_path
        self.rgb, self.luma, self.alpha = read_rgba_luma(input_path)
        self.annotation = white_annotation_mask(self.rgb)
        self.structure = structure_weight(self.luma, self.annotation) > 0.22
        self.bg = ~self.structure
        self.residual = build_wavelet_residual(self.luma, 6, 1, 4)
        self.height, self.width = self.luma.shape

    def compute_preview(self, roi: Roi, params: dict) -> dict:
        crop_residual = self.residual[roi.y0 : roi.y1, roi.x0 : roi.x1]
        crop_bg = self.bg[roi.y0 : roi.y1, roi.x0 : roi.x1]
        if crop_bg.mean() < 0.03:
            crop_bg = np.ones_like(crop_bg, dtype=bool)
        direction = params.get("direction", "up")
        horizontal = is_horizontal_direction(direction)
        fit_residual = crop_residual.T if horizontal else crop_residual
        fit_bg = crop_bg.T if horizontal else crop_bg
        fit_width = fit_residual.shape[1]
        center_origin = roi.y0 if horizontal else roi.x0

        if params["auto"]:
            comp = find_best_component(
                fit_residual,
                fit_bg,
                0,
                fit_width,
                params["curvature_range"],
                params["period_range"],
                params["center_steps"],
                params["curv_steps"],
                params["fine_steps"],
                (params["sign"],),
            )
            if comp is None or comp.score < params["min_score"]:
                raise RuntimeError("No reliable curve found in this ROI. Try manual mode or lower min_score.")
        else:
            comp = Component(
                0,
                fit_width,
                params["center_x"] - center_origin,
                params["curvature"],
                params["sign"],
                params["period"],
                0.0,
            )

        comp.y_shift = params["y_shifts"][0]
        comp.y_shifts = tuple(params["y_shifts"])
        comp.y_shift_weights = tuple(params["y_shift_weights"])

        fit_noise = fit_component(fit_residual, fit_bg, comp, params["harmonics"])
        noise_crop = fit_noise.T if horizontal else fit_noise
        support = np.ones_like(noise_crop, dtype=bool)
        corrected_crop, clipped_noise = apply_noise_correction(
            self.luma[roi.y0 : roi.y1, roi.x0 : roi.x1],
            noise_crop,
            support,
            params["alpha"],
            params["clip_sigma"],
            params["compensate_local_mean"],
            params["compensation_sigma"],
            params["compensation_strength"],
        )
        correction_crop = self.luma[roi.y0 : roi.y1, roi.x0 : roi.x1] - corrected_crop
        return {
            "family": params["family"],
            "roi": roi,
            "component": comp,
            "direction": direction,
            "params": params,
            "noise": clipped_noise.astype(np.float32),
            "correction": correction_crop.astype(np.float32),
            "weight": roi_weight(correction_crop.shape).astype(np.float32),
        }

    def merge_layers(self, layers: list[CorrectionLayer], preview: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
        family_sum: dict[str, np.ndarray] = {}
        family_weight: dict[str, np.ndarray] = {}
        support = np.zeros_like(self.luma, dtype=np.float32)

        active_items: list[tuple[str, Roi, np.ndarray, np.ndarray]] = []
        for layer in layers:
            if not layer.enabled:
                continue
            active_items.append((layer.family, layer.roi, np.load(layer.correction_path), np.load(layer.weight_path)))
        if preview is not None:
            active_items.append((preview["family"], preview["roi"], preview["correction"], preview["weight"]))

        for family, roi, correction_crop, weight_crop in active_items:
            if family not in family_sum:
                family_sum[family] = np.zeros_like(self.luma, dtype=np.float32)
                family_weight[family] = np.zeros_like(self.luma, dtype=np.float32)
            family_sum[family][roi.y0 : roi.y1, roi.x0 : roi.x1] += correction_crop * weight_crop
            family_weight[family][roi.y0 : roi.y1, roi.x0 : roi.x1] += weight_crop
            support[roi.y0 : roi.y1, roi.x0 : roi.x1] = np.maximum(support[roi.y0 : roi.y1, roi.x0 : roi.x1], weight_crop)

        total = np.zeros_like(self.luma, dtype=np.float32)
        for family, accum in family_sum.items():
            weight = family_weight[family]
            merged = np.zeros_like(self.luma, dtype=np.float32)
            valid = weight > 1e-6
            merged[valid] = accum[valid] / weight[valid]
            total += merged
        return total, support

    def combined_luma(self, layers: list[CorrectionLayer], preview: dict | None = None) -> np.ndarray:
        correction, _ = self.merge_layers(layers, preview)
        return np.clip(self.luma - correction, 0, 1)

    def save_layer_outputs(self, layer: CorrectionLayer, layers_dir: Path) -> None:
        roi = layer.roi
        correction = np.load(layer.correction_path)
        noise = np.load(layer.noise_path)
        corrected_crop = np.clip(self.luma[roi.y0 : roi.y1, roi.x0 : roi.x1] - correction, 0, 1)
        crop_rgb = self.rgb[roi.y0 : roi.y1, roi.x0 : roi.x1]
        crop_alpha = None if self.alpha is None else self.alpha[roi.y0 : roi.y1, roi.x0 : roi.x1]
        save_with_luma(layers_dir / f"correction_{layer.layer_id:03d}_roi_corrected.png", crop_rgb, corrected_crop, crop_alpha)
        save_gray(layers_dir / f"correction_{layer.layer_id:03d}_noise.png", noise, (0.2, 99.8))

    def export_final(self, layers: list[CorrectionLayer], output_path: Path) -> None:
        correction, support = self.merge_layers(layers, None)
        final_luma = np.clip(self.luma - correction, 0, 1)
        save_with_luma(output_path, self.rgb, final_luma, self.alpha)
        save_gray(output_path.with_name(output_path.stem + "_total_correction.png"), correction, (0.2, 99.8))
        imwrite_checked(
            output_path.with_name(output_path.stem + "_support.png"),
            (np.clip(support, 0, 1) * 255).astype(np.uint8),
        )
