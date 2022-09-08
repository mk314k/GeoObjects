"""
Circle (`Circle`) and axis-aligned/rotated ellipse (`Ellipse`) primitives.
"""
from __future__ import annotations
import math
from typing import Tuple, Union
import numpy as np
from geos.interfaces import GeoShape
from geos.point import Point
from geos.rect import Rect


class Circle(GeoShape):
    """2D circle defined by `center = (cx, cy)` and `radius >= 0`."""

    def __init__(
        self,
        center: Union[Point, Tuple[float, float]],
        radius: float,
    ) -> None:
        if radius < 0:
            raise ValueError("Circle radius must be non-negative")
        self.center = center if isinstance(center, Point) else Point(center[0], center[1])
        self.radius = float(radius)

    def area(self) -> float:
        return math.pi * self.radius * self.radius

    def perimeter(self) -> float:
        return 2.0 * math.pi * self.radius

    def bbox(self) -> Rect:
        r = self.radius
        return Rect(
            self.center.x - r,
            self.center.y - r,
            self.center.x + r,
            self.center.y + r,
        )

    def centroid(self) -> Point:
        return Point(self.center.x, self.center.y)

    def translate(self, dx: float, dy: float) -> Circle:
        return Circle(Point(self.center.x + dx, self.center.y + dy), self.radius)

    def scale(self, factor: float) -> Circle:
        return Circle(self.center, self.radius * abs(float(factor)))

    def is_overlapping(self, other: Union[Circle, Rect]) -> bool:
        """Check whether `self` intersects another `Circle` or axis-aligned `Rect`."""
        if isinstance(other, Circle):
            return self.center.distance_to(other.center) <= (self.radius + other.radius + 1e-9)
        if isinstance(other, Rect):
            cx = max(other.x1, min(self.center.x, other.x2))
            cy = max(other.y1, min(self.center.y, other.y2))
            return math.hypot(self.center.x - cx, self.center.y - cy) <= self.radius + 1e-9
        raise TypeError(f"Unsupported type for Circle.is_overlapping: {type(other).__name__}")

    def intersection_area(self, other: Circle) -> float:
        """
        Exact analytic area of the lens intersection between two circles `self` and `other`.
        """
        r1, r2 = self.radius, other.radius
        d = self.center.distance_to(other.center)
        if d >= r1 + r2:
            return 0.0
        if d <= abs(r1 - r2):
            r_min = min(r1, r2)
            return math.pi * r_min * r_min

        a1 = math.acos(max(-1.0, min(1.0, (d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1))))
        a2 = math.acos(max(-1.0, min(1.0, (d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2))))
        term = max(
            0.0,
            (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2),
        )
        return r1 * r1 * a1 + r2 * r2 * a2 - 0.5 * math.sqrt(term)

    def iou(self, other: Circle) -> float:
        inter = self.intersection_area(other)
        if inter == 0.0:
            return 0.0
        union = self.area() + other.area() - inter
        return inter / union if union > 0.0 else 0.0

    def sdf(self, pts: np.ndarray) -> np.ndarray:
        """Exact Euclidean SDF `||p - center||_2 - radius` on `pts` of shape `(..., 2)`."""
        pts_arr = np.asarray(pts, dtype=np.float64)
        dx = pts_arr[..., 0] - self.center.x
        dy = pts_arr[..., 1] - self.center.y
        return np.hypot(dx, dy) - self.radius

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Circle):
            return False
        return self.center == other.center and math.isclose(
            self.radius, other.radius, abs_tol=1e-9
        )

    def __repr__(self) -> str:
        return f"Circle(center={self.center}, radius={self.radius})"
