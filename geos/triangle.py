"""
2D triangle (`Triangle`) with barycentric coordinates, circumcircle/incircle,
and exact vectorized Euclidean SDF.
"""
from __future__ import annotations
import math
from typing import Tuple, Union
import numpy as np
from geos.circle import Circle
from geos.interfaces import GeoShape
from geos.point import Point
from geos.rect import Rect


class Triangle(GeoShape):
    """2D triangle defined by vertices `(a, b, c)`."""

    def __init__(
        self,
        a: Union[Point, Tuple[float, float]],
        b: Union[Point, Tuple[float, float]],
        c: Union[Point, Tuple[float, float]],
    ) -> None:
        self.a = a if isinstance(a, Point) else Point(a[0], a[1])
        self.b = b if isinstance(b, Point) else Point(b[0], b[1])
        self.c = c if isinstance(c, Point) else Point(c[0], c[1])

    def signed_area(self) -> float:
        """Signed area (`> 0` for counter-clockwise vertex ordering)."""
        return 0.5 * (self.b - self.a).cross(self.c - self.a)

    def area(self) -> float:
        return abs(self.signed_area())

    def perimeter(self) -> float:
        return (
            self.a.distance_to(self.b)
            + self.b.distance_to(self.c)
            + self.c.distance_to(self.a)
        )

    def bbox(self) -> Rect:
        xs = (self.a.x, self.b.x, self.c.x)
        ys = (self.a.y, self.b.y, self.c.y)
        return Rect(min(xs), min(ys), max(xs), max(ys))

    def centroid(self) -> Point:
        return Point(
            (self.a.x + self.b.x + self.c.x) / 3.0,
            (self.a.y + self.b.y + self.c.y) / 3.0,
        )

    def barycentric(
        self, pt: Union[Point, Tuple[float, float]]
    ) -> Tuple[float, float, float]:
        """Return barycentric coordinates `(u, v, w)` of `pt` with `u + v + w = 1`."""
        v0 = self.b - self.a
        v1 = self.c - self.a
        v2 = Point(float(pt[0]) - self.a.x, float(pt[1]) - self.a.y)
        denom = v0.cross(v1)
        if abs(denom) <= 1e-15:
            return (1.0, 0.0, 0.0)
        v = v2.cross(v1) / denom
        w = v0.cross(v2) / denom
        u = 1.0 - v - w
        return (u, v, w)

    def __repr__(self) -> str:
        return f"Triangle({self.a}, {self.b}, {self.c})"
