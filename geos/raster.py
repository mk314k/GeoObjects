"""
Vectorized NumPy coordinate grid generation, SDF rendering, boolean masks,
and anti-aliased fill/stroke evaluation.
"""
from typing import Optional, Tuple
import numpy as np
from geos.interfaces import GeoShape


def make_grid(
    height: int,
    width: int,
    bounds: Optional[Tuple[float, float, float, float]] = None,
) -> np.ndarray:
    """
    Build a `(height, width, 2)` coordinate grid `(x, y)` sampled at pixel centers.
    If `bounds = (x1, y1, x2, y2)` is omitted, uses `[0, width] x [0, height]`.
    """
    if bounds is None:
        x1, y1, x2, y2 = 0.0, 0.0, float(width), float(height)
    else:
        x1, y1, x2, y2 = (
            float(bounds[0]),
            float(bounds[1]),
            float(bounds[2]),
            float(bounds[3]),
        )
    xs = np.linspace(
        x1 + 0.5 * (x2 - x1) / max(1, width),
        x2 - 0.5 * (x2 - x1) / max(1, width),
        width,
        dtype=np.float64,
    )
    ys = np.linspace(
        y1 + 0.5 * (y2 - y1) / max(1, height),
        y2 - 0.5 * (y2 - y1) / max(1, height),
        height,
        dtype=np.float64,
    )
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx, gy], axis=-1)


def render_sdf(
    shape: GeoShape,
    height: int,
    width: int,
    bounds: Optional[Tuple[float, float, float, float]] = None,
) -> np.ndarray:
    """Evaluate `shape.sdf` over a `(height, width)` pixel grid."""
    grid = make_grid(height, width, bounds=bounds)
    return shape.sdf(grid)


def rasterize_mask(
    shape: GeoShape,
    height: int,
    width: int,
    bounds: Optional[Tuple[float, float, float, float]] = None,
) -> np.ndarray:
    """Return a `(height, width)` boolean mask where `shape.sdf <= 0`."""
    return render_sdf(shape, height, width, bounds=bounds) <= 0.0


def anti_aliased_fill(
    shape: GeoShape,
    height: int,
    width: int,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    edge_width: float = 1.0,
) -> np.ndarray:
    """
    Compute smooth `[0.0, 1.0]` coverage using a sub-pixel Hermite smoothstep
    over `[-edge_width/2, +edge_width/2]` around `sdf == 0`.
    """
    d = render_sdf(shape, height, width, bounds=bounds)
    w = max(1e-6, float(edge_width))
    t = np.clip(0.5 - d / w, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def stroke_mask(
    shape: GeoShape,
    stroke_width: float,
    height: int,
    width: int,
    bounds: Optional[Tuple[float, float, float, float]] = None,
) -> np.ndarray:
    """Return a `(height, width)` boolean outline mask where `|sdf| <= stroke_width / 2`."""
    d = render_sdf(shape, height, width, bounds=bounds)
    return np.abs(d) <= 0.5 * float(stroke_width)
