"""
Software architecture and diagramming shapes (`RoundedRect`, `Capsule`, `Diamond`,
`Parallelogram`, `Hexagon`, `RegularPolygon`, `Cylinder`, `CloudShape`, `Arrow`,
`OrthogonalConnector`).
"""
from __future__ import annotations
import math
from typing import Dict, List, Sequence, Tuple, Union
import numpy as np
from geos.circle import Circle, Ellipse
from geos.interfaces import GeoShape
from geos.point import Point, Segment
from geos.polygon import Polygon
from geos.rect import Rect
from geos.triangle import Triangle


class DiagramNode(GeoShape):
    """Mixin providing standard compass ports (`top`, `bottom`, `left`, `right`, `center`) for diagram connectors."""

    def ports(self) -> Dict[str, Point]:
        b = self.bbox()
        c = b.centroid()
        return {
            "center": c,
            "top": Point(c.x, b.y2),
            "bottom": Point(c.x, b.y1),
            "left": Point(b.x1, c.y),
            "right": Point(b.x2, c.y),
        }

    def port(self, name: str) -> Point:
        return self.ports()[name]


class RoundedRect(DiagramNode):
    """
    Axis-aligned rounded rectangle `[x1, x2] x [y1, y2]` with uniform or per-corner
    radii `(r_tl, r_tr, r_br, r_bl)` (service nodes, containers, UI cards).
    """

    def __init__(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        radius: Union[float, Tuple[float, float, float, float]] = 8.0,
    ) -> None:
        self.rect = Rect(x1, y1, x2, y2)
        max_r = 0.5 * min(self.rect.width(), self.rect.height())
        if isinstance(radius, (int, float)):
            r = max(0.0, min(float(radius), max_r))
            self.radii = (r, r, r, r)
        else:
            self.radii = tuple(max(0.0, min(float(v), max_r)) for v in radius[:4])  # type: ignore

    def area(self) -> float:
        r_sum_sq = sum(r * r for r in self.radii)
        return self.rect.area() - (1.0 - 0.25 * math.pi) * r_sum_sq

    def perimeter(self) -> float:
        r_sum = sum(self.radii)
        return self.rect.perimeter() - (2.0 - 0.5 * math.pi) * r_sum

    def bbox(self) -> Rect:
        return self.rect.bbox()

    def centroid(self) -> Point:
        return self.rect.centroid()

    def sdf(self, pts: np.ndarray) -> np.ndarray:
        pts_arr = np.asarray(pts, dtype=np.float64)
        c = self.rect.centroid()
        hx = 0.5 * self.rect.width()
        hy = 0.5 * self.rect.height()
        px = pts_arr[..., 0] - c.x
        py = pts_arr[..., 1] - c.y

        r_tl, r_tr, r_br, r_bl = self.radii
        r_right = np.where(py > 0.0, r_tr, r_br)
        r_left = np.where(py > 0.0, r_tl, r_bl)
        r = np.where(px > 0.0, r_right, r_left)

        qx = np.abs(px) - hx + r
        qy = np.abs(py) - hy + r
        outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
        inside = np.minimum(np.maximum(qx, qy), 0.0)
        return outside + inside - r


class Capsule(DiagramNode):
    """
    Stadium/pill shape formed by sweeping a circle of `radius` along segment `p1 -> p2`
    (terminal start/end nodes, message queues, badges).
    """

    def __init__(
        self,
        p1: Union[Point, Tuple[float, float]],
        p2: Union[Point, Tuple[float, float]],
        radius: float,
    ) -> None:
        if radius < 0:
            raise ValueError("Capsule radius must be non-negative")
        self.segment = Segment(p1, p2)
        self.radius = float(radius)

    def area(self) -> float:
        L = self.segment.length()
        r = self.radius
        return 2.0 * r * L + math.pi * r * r

    def perimeter(self) -> float:
        return 2.0 * self.segment.length() + 2.0 * math.pi * self.radius

    def bbox(self) -> Rect:
        r = self.radius
        x1 = min(self.segment.p1.x, self.segment.p2.x) - r
        y1 = min(self.segment.p1.y, self.segment.p2.y) - r
        x2 = max(self.segment.p1.x, self.segment.p2.x) + r
        y2 = max(self.segment.p1.y, self.segment.p2.y) + r
        return Rect(x1, y1, x2, y2)

    def centroid(self) -> Point:
        return self.segment.midpoint()

    def sdf(self, pts: np.ndarray) -> np.ndarray:
        return self.segment.sdf(pts, radius=self.radius)
