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

    def incircle(self) -> Circle:
        """Inscribed circle tangent to all three sides."""
        la = self.b.distance_to(self.c)
        lb = self.c.distance_to(self.a)
        lc = self.a.distance_to(self.b)
        p = la + lb + lc
        if p <= 1e-15:
            return Circle(self.a, 0.0)
        ix = (la * self.a.x + lb * self.b.x + lc * self.c.x) / p
        iy = (la * self.a.y + lb * self.b.y + lc * self.c.y) / p
        r = 2.0 * self.area() / p
        return Circle(Point(ix, iy), r)

    def circumcircle(self) -> Circle:
        """Circumscribed circle passing through vertices `a`, `b`, and `c`."""
        ax, ay = self.a.x, self.a.y
        bx, by = self.b.x - ax, self.b.y - ay
        cx, cy = self.c.x - ax, self.c.y - ay
        d = 2.0 * (bx * cy - by * cx)
        if abs(d) <= 1e-15:
            return Circle(self.centroid(), 0.0)
        b2 = bx * bx + by * by
        c2 = cx * cx + cy * cy
        ux = ax + (cy * b2 - by * c2) / d
        uy = ay + (bx * c2 - cx * b2) / d
        center = Point(ux, uy)
        return Circle(center, center.distance_to(self.a))

    def sdf(self, pts: np.ndarray) -> np.ndarray:
        """Exact vectorized Euclidean Signed Distance Field for a 2D triangle."""
        pts_arr = np.asarray(pts, dtype=np.float64)
        p0 = np.array([self.a.x, self.a.y], dtype=np.float64)
        p1 = np.array([self.b.x, self.b.y], dtype=np.float64)
        p2 = np.array([self.c.x, self.c.y], dtype=np.float64)

        e0 = p1 - p0
        e1 = p2 - p1
        e2 = p0 - p2

        v0 = pts_arr - p0
        v1 = pts_arr - p1
        v2 = pts_arr - p2

        def _edge_sq(v: np.ndarray, e: np.ndarray) -> np.ndarray:
            ee = max(float(np.dot(e, e)), 1e-18)
            h = np.clip(np.sum(v * e, axis=-1) / ee, 0.0, 1.0)
            pq = v - h[..., None] * e
            return np.sum(pq * pq, axis=-1)

        d_sq = np.minimum(
            np.minimum(_edge_sq(v0, e0), _edge_sq(v1, e1)),
            _edge_sq(v2, e2),
        )

        s = float(np.sign(e0[0] * e2[1] - e0[1] * e2[0]))
        if s == 0.0:
            return np.sqrt(d_sq)

        c0 = s * (v0[..., 0] * e0[1] - v0[..., 1] * e0[0])
        c1 = s * (v1[..., 0] * e1[1] - v1[..., 1] * e1[0])
        c2 = s * (v2[..., 0] * e2[1] - v2[..., 1] * e2[0])
        sign = np.where(np.minimum(np.minimum(c0, c1), c2) >= 0.0, -1.0, 1.0)
        return np.sqrt(d_sq) * sign

    def __repr__(self) -> str:
        return f"Triangle({self.a}, {self.b}, {self.c})"
