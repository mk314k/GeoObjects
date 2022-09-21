"""
2x2 linear matrix + translation transformations (`Transform2D`), inverse-mapped
shape wrappers (`TransformedShape`), and spatial domain warps (`GridRepeatShape`,
`RadialRepeatShape`, `TwistWarpShape`).
"""
from __future__ import annotations
import math
from typing import Optional, Sequence, Tuple, Union
import numpy as np
from geos.interfaces import GeoShape
from geos.point import Point
from geos.rect import Rect


class Transform2D:
    """
    2D affine transform `p' = A @ p + t`, where `A` is a 2x2 matrix and `t` is a
    2D translation vector.
    """

    def __init__(
        self,
        matrix: Optional[Union[Sequence[Sequence[float]], np.ndarray]] = None,
        offset: Union[Point, Tuple[float, float], np.ndarray] = (0.0, 0.0),
    ) -> None:
        if matrix is None:
            self.matrix = np.eye(2, dtype=np.float64)
        else:
            arr = np.asarray(matrix, dtype=np.float64)
            if arr.shape != (2, 2):
                raise ValueError(f"Transform2D matrix must be (2, 2), got {arr.shape}")
            self.matrix = arr.copy()
        self.offset = np.array([float(offset[0]), float(offset[1])], dtype=np.float64)

    @staticmethod
    def identity() -> Transform2D:
        return Transform2D()

    @staticmethod
    def translation(dx: float, dy: float) -> Transform2D:
        return Transform2D(np.eye(2, dtype=np.float64), (dx, dy))

    @staticmethod
    def scaling(
        sx: float,
        sy: Optional[float] = None,
        center: Union[Point, Tuple[float, float]] = (0.0, 0.0),
    ) -> Transform2D:
        if sy is None:
            sy = sx
        m = np.array([[float(sx), 0.0], [0.0, float(sy)]], dtype=np.float64)
        c = np.array([float(center[0]), float(center[1])], dtype=np.float64)
        t = c - m @ c
        return Transform2D(m, (t[0], t[1]))

    @staticmethod
    def rotation(
        angle_rad: float,
        center: Union[Point, Tuple[float, float]] = (0.0, 0.0),
    ) -> Transform2D:
        cs, sn = math.cos(angle_rad), math.sin(angle_rad)
        m = np.array([[cs, -sn], [sn, cs]], dtype=np.float64)
        c = np.array([float(center[0]), float(center[1])], dtype=np.float64)
        t = c - m @ c
        return Transform2D(m, (t[0], t[1]))

    @staticmethod
    def shear(
        shx: float = 0.0,
        shy: float = 0.0,
        center: Union[Point, Tuple[float, float]] = (0.0, 0.0),
    ) -> Transform2D:
        m = np.array([[1.0, float(shx)], [float(shy), 1.0]], dtype=np.float64)
        c = np.array([float(center[0]), float(center[1])], dtype=np.float64)
        t = c - m @ c
        return Transform2D(m, (t[0], t[1]))

    @staticmethod
    def reflection(
        axis: Union[Point, Tuple[float, float]] = (1.0, 0.0),
        center: Union[Point, Tuple[float, float]] = (0.0, 0.0),
    ) -> Transform2D:
        """Reflect across a line through `center` in direction `axis = (ux, uy)`."""
        ux, uy = float(axis[0]), float(axis[1])
        norm = math.hypot(ux, uy)
        if norm <= 1e-15:
            raise ValueError("Reflection axis must be non-zero")
        ux /= norm
        uy /= norm
        m = np.array(
            [
                [2.0 * ux * ux - 1.0, 2.0 * ux * uy],
                [2.0 * ux * uy, 2.0 * uy * uy - 1.0],
            ],
            dtype=np.float64,
        )
        c = np.array([float(center[0]), float(center[1])], dtype=np.float64)
        t = c - m @ c
        return Transform2D(m, (t[0], t[1]))

    def determinant(self) -> float:
        return float(
            self.matrix[0, 0] * self.matrix[1, 1]
            - self.matrix[0, 1] * self.matrix[1, 0]
        )

    def min_singular_value(self) -> float:
        """Smallest singular value `sigma_min(A)`, used for metric SDF scaling."""
        sv = np.linalg.svd(self.matrix, compute_uv=False)
        return float(sv[-1])

    def inverse(self) -> Transform2D:
        det = self.determinant()
        if abs(det) <= 1e-15:
            raise np.linalg.LinAlgError("Transform2D matrix is singular")
        inv_m = np.linalg.inv(self.matrix)
        inv_t = -(inv_m @ self.offset)
        return Transform2D(inv_m, (inv_t[0], inv_t[1]))

    def apply_point(self, pt: Union[Point, Tuple[float, float]]) -> Point:
        v = np.array([float(pt[0]), float(pt[1])], dtype=np.float64)
        res = self.matrix @ v + self.offset
        return Point(res[0], res[1])

    def apply(self, pts: np.ndarray) -> np.ndarray:
        """Apply transform to an array `pts` of shape `(..., 2)`."""
        pts_arr = np.asarray(pts, dtype=np.float64)
        return pts_arr @ self.matrix.T + self.offset

    def __matmul__(self, other: Transform2D) -> Transform2D:
        """Compose transforms `(self @ other)(p) = self(other(p))`."""
        m = self.matrix @ other.matrix
        t = self.matrix @ other.offset + self.offset
        return Transform2D(m, (t[0], t[1]))

    def __repr__(self) -> str:
        return f"Transform2D(matrix={self.matrix.tolist()}, offset={self.offset.tolist()})"


class TransformedShape(GeoShape):
    """
    Wraps any `GeoShape` under a 2D affine transformation `transform`.
    Queries map world coordinates `p` back to local coordinates via `transform.inverse()`.
    """

    def __init__(self, shape: GeoShape, transform: Transform2D) -> None:
        self.shape = shape
        self.transform = transform
        self._inv = transform.inverse()
        self._dist_scale = max(1e-9, transform.min_singular_value())
        self._area_scale = abs(transform.determinant())

    def area(self) -> float:
        return self.shape.area() * self._area_scale

    def perimeter(self) -> float:
        return self.shape.perimeter() * self._dist_scale

    def bbox(self) -> Rect:
        b = self.shape.bbox()
        corners = [self.transform.apply_point(pt) for pt in b.vertices()]
        xs = [c.x for c in corners]
        ys = [c.y for c in corners]
        return Rect(min(xs), min(ys), max(xs), max(ys))

    def centroid(self) -> Point:
        return self.transform.apply_point(self.shape.centroid())

    def sdf(self, pts: np.ndarray) -> np.ndarray:
        local_pts = self._inv.apply(pts)
        return self.shape.sdf(local_pts) * self._dist_scale

    def __repr__(self) -> str:
        return f"TransformedShape({self.shape!r}, {self.transform!r})"


class GridRepeatShape(GeoShape):
    """
    Spatial domain repetition wrapper that tiles `shape` across a 2D grid with
    spacing `(period_x, period_y)` inside optional bounding limits `bounds`.
    """

    def __init__(
        self,
        shape: GeoShape,
        period_x: float,
        period_y: float,
        bounds: Optional[Rect] = None,
    ) -> None:
        if period_x <= 0 or period_y <= 0:
            raise ValueError("Grid periods must be positive")
        self.shape = shape
        self.period_x = float(period_x)
        self.period_y = float(period_y)
        self.bounds = bounds if bounds is not None else shape.bbox()

    def area(self) -> float:
        return self.shape.area()

    def perimeter(self) -> float:
        return self.shape.perimeter()

    def bbox(self) -> Rect:
        return self.bounds

    def centroid(self) -> Point:
        return self.bounds.centroid()

    def sdf(self, pts: np.ndarray) -> np.ndarray:
        pts_arr = np.asarray(pts, dtype=np.float64)
        c = self.shape.centroid()
        px, py = self.period_x, self.period_y
        # Fold coordinates into the fundamental cell centered at c
        dx = pts_arr[..., 0] - c.x
        dy = pts_arr[..., 1] - c.y
        folded_x = c.x + (dx - px * np.round(dx / px))
        folded_y = c.y + (dy - py * np.round(dy / py))
        folded = np.stack([folded_x, folded_y], axis=-1)
        d_tile = self.shape.sdf(folded)
        if self.bounds is not None:
            return np.maximum(d_tile, self.bounds.sdf(pts_arr))
        return d_tile


class RadialRepeatShape(GeoShape):
    """
    Angular domain repetition wrapper that replicates `shape` `count` times
    symmetrically around `center`.
    """

    def __init__(
        self,
        shape: GeoShape,
        count: int,
        center: Union[Point, Tuple[float, float]] = (0.0, 0.0),
    ) -> None:
        if count < 1:
            raise ValueError("Radial repeat count must be >= 1")
        self.shape = shape
        self.count = int(count)
        self.center = center if isinstance(center, Point) else Point(center[0], center[1])

    def area(self) -> float:
        return self.shape.area() * self.count

    def perimeter(self) -> float:
        return self.shape.perimeter() * self.count

    def bbox(self) -> Rect:
        b = self.shape.bbox()
        max_r = max(self.center.distance_to(v) for v in b.vertices())
        return Rect(
            self.center.x - max_r,
            self.center.y - max_r,
            self.center.x + max_r,
            self.center.y + max_r,
        )

    def centroid(self) -> Point:
        return Point(self.center.x, self.center.y)

    def sdf(self, pts: np.ndarray) -> np.ndarray:
        pts_arr = np.asarray(pts, dtype=np.float64)
        dx = pts_arr[..., 0] - self.center.x
        dy = pts_arr[..., 1] - self.center.y
        r = np.hypot(dx, dy)
        theta = np.arctan2(dy, dx)
        sector = 2.0 * math.pi / self.count
        folded_theta = theta - sector * np.round(theta / sector)
        fx = self.center.x + r * np.cos(folded_theta)
        fy = self.center.y + r * np.sin(folded_theta)
        return self.shape.sdf(np.stack([fx, fy], axis=-1))


class TwistWarpShape(GeoShape):
    """
    Non-linear space warp that twists `shape` around `center` by `strength`
    radians per unit distance along $Y$ (or radial distance).
    """

    def __init__(
        self,
        shape: GeoShape,
        strength: float,
        center: Union[Point, Tuple[float, float]] = (0.0, 0.0),
    ) -> None:
        self.shape = shape
        self.strength = float(strength)
        self.center = center if isinstance(center, Point) else Point(center[0], center[1])

    def area(self) -> float:
        return self.shape.area()

    def perimeter(self) -> float:
        return self.shape.perimeter()

    def bbox(self) -> Rect:
        b = self.shape.bbox()
        max_r = max(self.center.distance_to(v) for v in b.vertices())
        return Rect(
            self.center.x - max_r,
            self.center.y - max_r,
            self.center.x + max_r,
            self.center.y + max_r,
        )

    def centroid(self) -> Point:
        return Point(self.center.x, self.center.y)

    def sdf(self, pts: np.ndarray) -> np.ndarray:
        pts_arr = np.asarray(pts, dtype=np.float64)
        dx = pts_arr[..., 0] - self.center.x
        dy = pts_arr[..., 1] - self.center.y
        r = np.hypot(dx, dy)
        ang = -self.strength * r
        cs, sn = np.cos(ang), np.sin(ang)
        wx = self.center.x + cs * dx - sn * dy
        wy = self.center.y + sn * dx + cs * dy
        return self.shape.sdf(np.stack([wx, wy], axis=-1))
