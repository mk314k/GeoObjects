"""
Vector/SDF styling, gradients, drop shadows, outer glows, 2.5D bevel lighting,
and multi-layer RGBA compositing (`Canvas`).
"""
from __future__ import annotations
import math
from typing import Optional, Sequence, Tuple, Union
import numpy as np
from geos.convolution import gaussian_blur, sobel_gradients
from geos.interfaces import GeoShape
from geos.point import Point
from geos.raster import make_grid

RGBA = Tuple[float, float, float, float]
ColorInput = Union[Tuple[float, ...], Sequence[float], str]


def parse_color(col: ColorInput) -> np.ndarray:
    """
    Normalize a color specification (`(r, g, b)`, `(r, g, b, a)` in `[0, 1]` or
    `[0, 255]`, or hex string `'#RRGGBB'` / `'#RRGGBBAA'`) into a 4-float RGBA
    array in `[0.0, 1.0]`.
    """
    if isinstance(col, str):
        s = col.strip().lstrip("#")
        if len(s) == 6:
            r = int(s[0:2], 16) / 255.0
            g = int(s[2:4], 16) / 255.0
            b = int(s[4:6], 16) / 255.0
            return np.array([r, g, b, 1.0], dtype=np.float64)
        if len(s) == 8:
            r = int(s[0:2], 16) / 255.0
            g = int(s[2:4], 16) / 255.0
            b = int(s[4:6], 16) / 255.0
            a = int(s[6:8], 16) / 255.0
            return np.array([r, g, b, a], dtype=np.float64)
        raise ValueError(f"Unsupported hex color string: {col}")

    vals = [float(v) for v in col]
    if len(vals) not in (3, 4):
        raise ValueError("Color tuple must have 3 (RGB) or 4 (RGBA) values")
    scale = 255.0 if any(v > 1.0 for v in vals[:3]) else 1.0
    r, g, b = vals[0] / scale, vals[1] / scale, vals[2] / scale
    a = (vals[3] / scale if vals[3] > 1.0 else vals[3]) if len(vals) == 4 else 1.0
    return np.clip(np.array([r, g, b, a], dtype=np.float64), 0.0, 1.0)


def smooth_coverage(sdf_grid: np.ndarray, aa_width: float = 1.2) -> np.ndarray:
    """Sub-pixel Hermite smoothstep coverage in `[0.0, 1.0]` for `sdf <= 0`."""
    w = max(1e-6, float(aa_width))
    t = np.clip(0.5 - sdf_grid / w, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def alpha_over(dst: np.ndarray, src: np.ndarray) -> np.ndarray:
    """
    Porter-Duff `src over dst` alpha compositing on `(H, W, 4)` RGBA float buffers
    in `[0.0, 1.0]`.
    """
    sa = src[..., 3:4]
    da = dst[..., 3:4]
    out_a = sa + da * (1.0 - sa)
    safe_a = np.where(out_a > 1e-12, out_a, 1.0)
    out_rgb = (src[..., :3] * sa + dst[..., :3] * da * (1.0 - sa)) / safe_a
    out_rgb = np.where(out_a > 1e-12, out_rgb, 0.0)
    return np.concatenate([np.clip(out_rgb, 0.0, 1.0), np.clip(out_a, 0.0, 1.0)], axis=-1)


class FillStyle:
    """Base interface for evaluating an `(H, W, 4)` RGBA color field over a coordinate grid."""

    def evaluate(self, grid: np.ndarray, sdf_grid: np.ndarray = None) -> np.ndarray:
        raise NotImplementedError


class SolidFill(FillStyle):
    """Uniform RGBA color fill."""

    def __init__(self, color: ColorInput) -> None:
        self.rgba = parse_color(color)

    def evaluate(self, grid: np.ndarray, sdf_grid: np.ndarray = None) -> np.ndarray:
        h, w = grid.shape[:2]
        out = np.empty((h, w, 4), dtype=np.float64)
        out[...] = self.rgba
        return out


class LinearGradient(FillStyle):
    """Linear color interpolation along the segment from `p1` to `p2`."""

    def __init__(
        self,
        p1: Union[Point, Tuple[float, float]],
        p2: Union[Point, Tuple[float, float]],
        start_color: ColorInput,
        end_color: ColorInput,
    ) -> None:
        self.p1 = np.array([float(p1[0]), float(p1[1])], dtype=np.float64)
        self.p2 = np.array([float(p2[0]), float(p2[1])], dtype=np.float64)
        self.c1 = parse_color(start_color)
        self.c2 = parse_color(end_color)

    def evaluate(self, grid: np.ndarray, sdf_grid: np.ndarray = None) -> np.ndarray:
        v = self.p2 - self.p1
        denom = max(1e-15, float(np.dot(v, v)))
        t = np.clip(np.sum((grid - self.p1) * v, axis=-1, keepdims=True) / denom, 0.0, 1.0)
        return (1.0 - t) * self.c1 + t * self.c2


class RadialGradient(FillStyle):
    """Radial color interpolation from `center` out to `radius`."""

    def __init__(
        self,
        center: Union[Point, Tuple[float, float]],
        radius: float,
        inner_color: ColorInput,
        outer_color: ColorInput,
    ) -> None:
        self.center = np.array([float(center[0]), float(center[1])], dtype=np.float64)
        self.radius = max(1e-6, float(radius))
        self.c_in = parse_color(inner_color)
        self.c_out = parse_color(outer_color)

    def evaluate(self, grid: np.ndarray, sdf_grid: np.ndarray = None) -> np.ndarray:
        dist = np.linalg.norm(grid - self.center, axis=-1, keepdims=True)
        t = np.clip(dist / self.radius, 0.0, 1.0)
        return (1.0 - t) * self.c_in + t * self.c_out


class StrokeStyle:
    """
    Boundary stroke specification with `width`, `color`, and `align`
    (`'center'`, `'inside'`, or `'outside'`).
    """

    def __init__(
        self,
        width: float = 2.0,
        color: ColorInput = (0.1, 0.1, 0.1, 1.0),
        align: str = "center",
    ) -> None:
        if align not in ("center", "inside", "outside"):
            raise ValueError("StrokeStyle align must be 'center', 'inside', or 'outside'")
        self.width = max(0.0, float(width))
        self.rgba = parse_color(color)
        self.align = align

    def coverage(self, sdf_grid: np.ndarray, aa_width: float = 1.2) -> np.ndarray:
        hw = 0.5 * self.width
        if self.align == "center":
            stroke_sdf = np.abs(sdf_grid) - hw
        elif self.align == "inside":
            stroke_sdf = np.abs(sdf_grid + hw) - hw
        else:
            stroke_sdf = np.abs(sdf_grid - hw) - hw
        return smooth_coverage(stroke_sdf, aa_width=aa_width)
