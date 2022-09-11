"""
Simple 2D polygon (`Polygon`) supporting shoelace area, convex hull, Sutherland-Hodgman
polygon clipping, ear-clipping triangulation, and vectorized NumPy SDF evaluation.
"""
from __future__ import annotations
from typing import List, Sequence, Tuple, Union
import numpy as np
from geos.interfaces import GeoShape
from geos.point import Point
from geos.rect import Rect
from geos.triangle import Triangle


class Polygon(GeoShape):
    """2D simple polygon specified by an ordered sequence of vertices."""

    def __init__(
        self, vertices: Sequence[Union[Point, Tuple[float, float]]]
    ) -> None:
        if len(vertices) < 3:
            raise ValueError("Polygon requires at least 3 vertices")
        self.vertices: List[Point] = [
            v if isinstance(v, Point) else Point(v[0], v[1]) for v in vertices
        ]

    def to_numpy(self) -> np.ndarray:
        return np.array([[v.x, v.y] for v in self.vertices], dtype=np.float64)

    def signed_area(self) -> float:
        """Shoelace formula signed area (`> 0` for counter-clockwise orientation)."""
        pts = self.vertices
        n = len(pts)
        acc = 0.0
        for i in range(n):
            p1 = pts[i]
            p2 = pts[(i + 1) % n]
            acc += p1.x * p2.y - p2.x * p1.y
        return 0.5 * acc

    def area(self) -> float:
        return abs(self.signed_area())

    def perimeter(self) -> float:
        pts = self.vertices
        n = len(pts)
        return sum(pts[i].distance_to(pts[(i + 1) % n]) for i in range(n))

    def bbox(self) -> Rect:
        xs = [v.x for v in self.vertices]
        ys = [v.y for v in self.vertices]
        return Rect(min(xs), min(ys), max(xs), max(ys))

    def centroid(self) -> Point:
        """Area-weighted polygon centroid."""
        pts = self.vertices
        n = len(pts)
        a6 = 6.0 * self.signed_area()
        if abs(a6) <= 1e-15:
            return Point(
                sum(v.x for v in pts) / n,
                sum(v.y for v in pts) / n,
            )
        cx = 0.0
        cy = 0.0
        for i in range(n):
            p1 = pts[i]
            p2 = pts[(i + 1) % n]
            cross = p1.x * p2.y - p2.x * p1.y
            cx += (p1.x + p2.x) * cross
            cy += (p1.y + p2.y) * cross
        return Point(cx / a6, cy / a6)

    def ensure_ccw(self) -> Polygon:
        """Return a copy of the polygon with counter-clockwise vertex ordering."""
        if self.signed_area() < 0.0:
            return Polygon(list(reversed(self.vertices)))
        return Polygon(self.vertices)

    def __repr__(self) -> str:
        return f"Polygon({self.vertices})"
