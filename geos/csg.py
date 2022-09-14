"""
Constructive Solid Geometry (CSG) expression nodes for combining `GeoShape` objects.
"""
from __future__ import annotations
import numpy as np
from geos.interfaces import GeoShape
from geos.point import Point
from geos.rect import Rect


class CSGNode(GeoShape):
    """Base binary CSG node combining `left` and `right` shapes."""

    def __init__(self, left: GeoShape, right: GeoShape) -> None:
        self.left = left
        self.right = right

    def bbox(self) -> Rect:
        return self.left.bbox().bounding_union(self.right.bbox())

    def centroid(self) -> Point:
        return self.bbox().centroid()

    def estimate_area(self, resolution: int = 256) -> float:
        """Grid-integrated area estimate for arbitrary CSG combinations."""
        box = self.bbox()
        if box.area() <= 0.0:
            return 0.0
        mask = self.to_mask(
            resolution,
            resolution,
            bounds=(box.x1, box.y1, box.x2, box.y2),
        )
        return float(np.mean(mask)) * box.area()

    def area(self) -> float:
        return self.estimate_area()


class UnionShape(CSGNode):
    """Set union `A | B` with SDF `min(d_A, d_B)`."""

    def sdf(self, pts: np.ndarray) -> np.ndarray:
        return np.minimum(self.left.sdf(pts), self.right.sdf(pts))

    def __repr__(self) -> str:
        return f"UnionShape({self.left!r}, {self.right!r})"


class IntersectionShape(CSGNode):
    """Set intersection `A & B` with SDF `max(d_A, d_B)`."""

    def bbox(self) -> Rect:
        inter = self.left.bbox().intersection(self.right.bbox())
        return inter if inter is not None else Rect(0.0, 0.0, 0.0, 0.0)

    def sdf(self, pts: np.ndarray) -> np.ndarray:
        return np.maximum(self.left.sdf(pts), self.right.sdf(pts))

    def __repr__(self) -> str:
        return f"IntersectionShape({self.left!r}, {self.right!r})"


class DifferenceShape(CSGNode):
    """Set difference `A - B` with SDF `max(d_A, -d_B)`."""

    def bbox(self) -> Rect:
        return self.left.bbox()

    def sdf(self, pts: np.ndarray) -> np.ndarray:
        return np.maximum(self.left.sdf(pts), -self.right.sdf(pts))

    def __repr__(self) -> str:
        return f"DifferenceShape({self.left!r}, {self.right!r})"


class SymmetricDifferenceShape(CSGNode):
    """Symmetric difference (XOR) `A ^ B = (A | B) - (A & B)`."""

    def sdf(self, pts: np.ndarray) -> np.ndarray:
        da = self.left.sdf(pts)
        db = self.right.sdf(pts)
        return np.maximum(np.minimum(da, db), -np.maximum(da, db))

    def __repr__(self) -> str:
        return f"SymmetricDifferenceShape({self.left!r}, {self.right!r})"


class SmoothUnionShape(CSGNode):
    """
    Polynomial smooth-minimum union `smin(d_A, d_B, k)` for organic blending and
    procedural texture generation.
    """

    def __init__(self, left: GeoShape, right: GeoShape, k: float = 5.0) -> None:
        super().__init__(left, right)
        self.k = max(1e-6, float(k))

    def sdf(self, pts: np.ndarray) -> np.ndarray:
        da = self.left.sdf(pts)
        db = self.right.sdf(pts)
        h = np.clip(0.5 + 0.5 * (db - da) / self.k, 0.0, 1.0)
        return db * (1.0 - h) + da * h - self.k * h * (1.0 - h)

    def __repr__(self) -> str:
        return f"SmoothUnionShape({self.left!r}, {self.right!r}, k={self.k})"
