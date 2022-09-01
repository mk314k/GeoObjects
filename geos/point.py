"""
2D point/vector (`Point`) and line segment (`Segment`) primitives.
"""
from __future__ import annotations
import math
from typing import Optional, Tuple, Union
import numpy as np


class Point:
    """2D Cartesian coordinate or vector `(x, y)`."""

    def __init__(self, x: float, y: float) -> None:
        self.__x = float(x)
        self.__y = float(y)

    @property
    def x(self) -> float:
        return self.__x

    @property
    def y(self) -> float:
        return self.__y

    def to_numpy(self) -> np.ndarray:
        return np.array([self.__x, self.__y], dtype=np.float64)

    def __getitem__(self, index: int) -> float:
        if index == 0:
            return self.__x
        if index == 1:
            return self.__y
        raise IndexError(f"Point index {index} out of range (expected 0 or 1)")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (Point, tuple, list, np.ndarray)) and len(other) >= 2:
            return (
                math.isclose(self.__x, float(other[0]), abs_tol=1e-9)
                and math.isclose(self.__y, float(other[1]), abs_tol=1e-9)
            )
        return False

    def __len__(self) -> int:
        return 2

    def __str__(self) -> str:
        return str((self.__x, self.__y))

    def __repr__(self) -> str:
        return f"Point({self.__x}, {self.__y})"

    def __add__(self, point: Union[Point, tuple, list, np.ndarray]) -> Point:
        return Point(self.__x + float(point[0]), self.__y + float(point[1]))

    def __radd__(self, point: Union[Point, tuple, list, np.ndarray]) -> Point:
        return Point(self.__x + float(point[0]), self.__y + float(point[1]))

    def __sub__(self, point: Union[Point, tuple, list, np.ndarray]) -> Point:
        return Point(self.__x - float(point[0]), self.__y - float(point[1]))

    def __mul__(self, scalar: float) -> Point:
        return Point(self.__x * float(scalar), self.__y * float(scalar))

    def __rmul__(self, scalar: float) -> Point:
        return Point(self.__x * float(scalar), self.__y * float(scalar))

    def dot(self, other: Union[Point, tuple, list]) -> float:
        """2D dot product `x1*x2 + y1*y2`."""
        return self.__x * float(other[0]) + self.__y * float(other[1])

    def cross(self, other: Union[Point, tuple, list]) -> float:
        """2D scalar cross product (z-component) `x1*y2 - y1*x2`."""
        return self.__x * float(other[1]) - self.__y * float(other[0])

    def norm(self) -> float:
        """Euclidean length `hypot(x, y)`."""
        return math.hypot(self.__x, self.__y)

    def distance_to(self, other: Union[Point, tuple, list]) -> float:
        """Euclidean distance between `self` and `other`."""
        return math.hypot(self.__x - float(other[0]), self.__y - float(other[1]))

    def rotate(
        self, angle_rad: float, origin: Union[Point, Tuple[float, float]] = (0.0, 0.0)
    ) -> Point:
        """Rotate counter-clockwise by `angle_rad` radians around `origin`."""
        ox, oy = float(origin[0]), float(origin[1])
        dx, dy = self.__x - ox, self.__y - oy
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        return Point(ox + c * dx - s * dy, oy + s * dx + c * dy)

    def lerp(self, other: Union[Point, Tuple[float, float]], t: float) -> Point:
        """Linear interpolation `(1 - t) * self + t * other`."""
        return Point(
            self.__x + float(t) * (float(other[0]) - self.__x),
            self.__y + float(t) * (float(other[1]) - self.__y),
        )

    def is_upper_right_to(self, point: Point) -> bool:
        """Return True if `self` lies at or above/right of `point`."""
        return self.__x >= point.__x and self.__y >= point.__y

    def min_xy(self, point: Point) -> Point:
        """Return componentwise minimum `(min(x1, x2), min(y1, y2))`."""
        return Point(min(self.__x, point.__x), min(self.__y, point.__y))

    def max_xy(self, point: Point) -> Point:
        """Return componentwise maximum `(max(x1, x2), max(y1, y2))`."""
        return Point(max(self.__x, point.__x), max(self.__y, point.__y))

    isUpperRightTo = is_upper_right_to
    minXYCompare = min_xy
    maxXYCompare = max_xy


class Segment:
    """Directed 2D line segment from `p1` to `p2`."""

    def __init__(
        self,
        p1: Union[Point, Tuple[float, float]],
        p2: Union[Point, Tuple[float, float]],
    ) -> None:
        self.p1 = p1 if isinstance(p1, Point) else Point(p1[0], p1[1])
        self.p2 = p2 if isinstance(p2, Point) else Point(p2[0], p2[1])

    def length(self) -> float:
        return self.p1.distance_to(self.p2)

    def midpoint(self) -> Point:
        return self.p1.lerp(self.p2, 0.5)

    def closest_point(self, pt: Union[Point, Tuple[float, float]]) -> Point:
        """Project `pt` onto the segment, clamped to endpoints `[p1, p2]`."""
        ab = self.p2 - self.p1
        denom = ab.dot(ab)
        if denom <= 1e-18:
            return self.p1
        ap = Point(pt[0] - self.p1.x, pt[1] - self.p1.y)
        t = max(0.0, min(1.0, ap.dot(ab) / denom))
        return self.p1.lerp(self.p2, t)

    def distance_to_point(self, pt: Union[Point, Tuple[float, float]]) -> float:
        """Minimum Euclidean distance from `pt` to any point on this segment."""
        return self.closest_point(pt).distance_to(pt)

    def intersection(self, other: Segment) -> Optional[Point]:
        """
        Compute the intersection point with `other` if the two segments cross;
        return `None` if parallel or disjoint.
        """
        r = self.p2 - self.p1
        s = other.p2 - other.p1
        rxs = r.cross(s)
        qp = other.p1 - self.p1
        if abs(rxs) <= 1e-12:
            return None
        t = qp.cross(s) / rxs
        u = qp.cross(r) / rxs
        if -1e-9 <= t <= 1.0 + 1e-9 and -1e-9 <= u <= 1.0 + 1e-9:
            return self.p1.lerp(self.p2, max(0.0, min(1.0, t)))
        return None

    def sdf(self, pts: np.ndarray, radius: float = 0.0) -> np.ndarray:
        """
        Vectorized capsule/segment distance field for an array `pts` of shape `(..., 2)`.
        Subtract `radius > 0` to obtain a thick capsule SDF.
        """
        pts_arr = np.asarray(pts, dtype=np.float64)
        a = np.array([self.p1.x, self.p1.y], dtype=np.float64)
        b = np.array([self.p2.x, self.p2.y], dtype=np.float64)
        pa = pts_arr - a
        ba = b - a
        denom = float(np.dot(ba, ba))
        if denom <= 1e-18:
            return np.linalg.norm(pa, axis=-1) - radius
        h = np.clip(np.sum(pa * ba, axis=-1) / denom, 0.0, 1.0)
        diff = pa - h[..., None] * ba
        return np.linalg.norm(diff, axis=-1) - radius
