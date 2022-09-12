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

    @staticmethod
    def convex_hull(
        points: Sequence[Union[Point, Tuple[float, float]]]
    ) -> Polygon:
        """Andrew's Monotone Chain `O(N log N)` convex hull algorithm."""
        pts = sorted(
            {(float(p[0]), float(p[1])) for p in points}
        )
        if len(pts) < 3:
            raise ValueError("Convex hull requires at least 3 distinct points")

        def _cross(
            o: Tuple[float, float],
            a: Tuple[float, float],
            b: Tuple[float, float],
        ) -> float:
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower: List[Tuple[float, float]] = []
        for p in pts:
            while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0.0:
                lower.pop()
            lower.append(p)

        upper: List[Tuple[float, float]] = []
        for p in reversed(pts):
            while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0.0:
                upper.pop()
            upper.append(p)

        hull_pts = lower[:-1] + upper[:-1]
        return Polygon([Point(x, y) for x, y in hull_pts])

    def clip_with_convex(self, clip_poly: Polygon) -> List[Point]:
        """
        Sutherland-Hodgman polygon clipping: intersect `self` against convex polygon
        `clip_poly` and return the clipped vertex list (empty if disjoint).
        """
        subject = [Point(v.x, v.y) for v in self.ensure_ccw().vertices]
        clipper = clip_poly.ensure_ccw().vertices
        m = len(clipper)

        def _inside(p: Point, a: Point, b: Point) -> bool:
            return (b - a).cross(p - a) >= -1e-9

        def _line_intersect(p1: Point, p2: Point, a: Point, b: Point) -> Point:
            r = p2 - p1
            s = b - a
            rxs = r.cross(s)
            if abs(rxs) <= 1e-15:
                return p2
            t = (a - p1).cross(s) / rxs
            return p1.lerp(p2, t)

        output = subject
        for i in range(m):
            a = clipper[i]
            b = clipper[(i + 1) % m]
            input_list = output
            output = []
            if not input_list:
                break
            s_pt = input_list[-1]
            for e_pt in input_list:
                if _inside(e_pt, a, b):
                    if not _inside(s_pt, a, b):
                        output.append(_line_intersect(s_pt, e_pt, a, b))
                    output.append(e_pt)
                elif _inside(s_pt, a, b):
                    output.append(_line_intersect(s_pt, e_pt, a, b))
                s_pt = e_pt
        return output

    def __repr__(self) -> str:
        return f"Polygon({self.vertices})"
