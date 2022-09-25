"""
Vectorized 2D and separable 1D-1D NumPy convolutions, Gaussian/box blurs,
Sobel surface-normal gradients, and morphological SDF offset wrappers.
"""
from __future__ import annotations
import math
from typing import Tuple
import numpy as np
from geos.interfaces import GeoShape
from geos.point import Point
from geos.rect import Rect


def convolve1d_axis(
    arr: np.ndarray,
    kernel_1d: np.ndarray,
    axis: int = 0,
    pad_mode: str = "edge",
) -> np.ndarray:
    """
    Vectorized 1D correlation/convolution along `axis` (`0` for rows/vertical,
    `1` for cols/horizontal) on a 2D `(H, W)` or 3D `(H, W, C)` array.
    """
    x = np.asarray(arr, dtype=np.float64)
    k = np.asarray(kernel_1d, dtype=np.float64).ravel()
    k_len = len(k)
    if k_len == 0:
        return x.copy()
    pad_before = k_len // 2
    pad_after = k_len - 1 - pad_before

    pad_spec = [(0, 0)] * x.ndim
    pad_spec[axis] = (pad_before, pad_after)
    padded = np.pad(x, pad_spec, mode=pad_mode)

    out = np.zeros_like(x, dtype=np.float64)
    n = x.shape[axis]
    for i, weight in enumerate(k):
        if weight == 0.0:
            continue
        slc = [slice(None)] * x.ndim
        slc[axis] = slice(i, i + n)
        out += weight * padded[tuple(slc)]
    return out


def convolve_separable(
    arr: np.ndarray,
    kernel_x: np.ndarray,
    kernel_y: np.ndarray,
    pad_mode: str = "edge",
) -> np.ndarray:
    """Apply a separable 2D filter in $O(K_x + K_y)$ per pixel."""
    temp = convolve1d_axis(arr, kernel_x, axis=1, pad_mode=pad_mode)
    return convolve1d_axis(temp, kernel_y, axis=0, pad_mode=pad_mode)


def convolve2d(
    arr: np.ndarray,
    kernel_2d: np.ndarray,
    pad_mode: str = "edge",
) -> np.ndarray:
    """
    Apply an arbitrary 2D `(Kh, Kw)` convolution kernel to a 2D `(H, W)` or
    3D `(H, W, C)` array using vectorized shifted accumulation.
    """
    x = np.asarray(arr, dtype=np.float64)
    k = np.asarray(kernel_2d, dtype=np.float64)
    if k.ndim != 2:
        raise ValueError("kernel_2d must be a 2D array")
    kh, kw = k.shape
    ph0, ph1 = kh // 2, kh - 1 - (kh // 2)
    pw0, pw1 = kw // 2, kw - 1 - (kw // 2)

    pad_spec = [(ph0, ph1), (pw0, pw1)] + [(0, 0)] * (x.ndim - 2)
    padded = np.pad(x, pad_spec, mode=pad_mode)

    h, w = x.shape[:2]
    out = np.zeros_like(x, dtype=np.float64)
    for r in range(kh):
        for c in range(kw):
            weight = float(k[r, c])
            if weight == 0.0:
                continue
            out += weight * padded[r : r + h, c : c + w]
    return out


def gaussian_kernel_1d(sigma: float, radius: int = None) -> np.ndarray:
    """Generate a normalized 1D Gaussian kernel with standard deviation `sigma`."""
    if sigma <= 0.0:
        return np.array([1.0], dtype=np.float64)
    if radius is None:
        radius = max(1, int(math.ceil(3.0 * sigma)))
    xs = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (xs / sigma) ** 2)
    return k / np.sum(k)


def gaussian_blur(
    arr: np.ndarray, sigma: float, pad_mode: str = "edge"
) -> np.ndarray:
    """Separable 2D Gaussian blur on a 2D field or 3D `(H, W, C)` image."""
    if sigma <= 0.0:
        return np.asarray(arr, dtype=np.float64).copy()
    k = gaussian_kernel_1d(sigma)
    return convolve_separable(arr, k, k, pad_mode=pad_mode)


def box_blur(arr: np.ndarray, radius: int, pad_mode: str = "edge") -> np.ndarray:
    """Separable 2D uniform box filter of window `(2*radius + 1) x (2*radius + 1)`."""
    r = max(0, int(radius))
    if r == 0:
        return np.asarray(arr, dtype=np.float64).copy()
    size = 2 * r + 1
    k = np.full(size, 1.0 / size, dtype=np.float64)
    return convolve_separable(arr, k, k, pad_mode=pad_mode)


def sobel_gradients(field: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 2D Sobel partial derivatives `(df_dx, df_dy)` on a scalar field `(H, W)`
    (e.g., an SDF grid to extract unit surface normals for 2.5D bevel shading).
    """
    deriv = np.array([-0.5, 0.0, 0.5], dtype=np.float64)
    smooth = np.array([0.25, 0.5, 0.25], dtype=np.float64)
    gx = convolve_separable(field, deriv, smooth, pad_mode="edge")
    gy = convolve_separable(field, smooth, deriv, pad_mode="edge")
    return gx, gy


def sharpen(
    arr: np.ndarray, amount: float = 1.0, sigma: float = 1.0
) -> np.ndarray:
    """Unsharp mask filter `arr + amount * (arr - gaussian_blur(arr, sigma))`."""
    x = np.asarray(arr, dtype=np.float64)
    blurred = gaussian_blur(x, sigma=sigma)
    return x + float(amount) * (x - blurred)
