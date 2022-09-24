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


class Diamond(DiagramNode):
    """
    Axis-aligned rhombus centered at `center` with horizontal half-width `rx` and
    vertical half-height `ry` (decision / gateway nodes).
    """

    def __init__(
        self,
        center: Union[Point, Tuple[float, float]],
        rx: float,
        ry: float,
    ) -> None:
        if rx <= 0 or ry <= 0:
            raise ValueError("Diamond half-extents rx and ry must be positive")
        self.center = center if isinstance(center, Point) else Point(center[0], center[1])
        self.rx = float(rx)
        self.ry = float(ry)
        self._poly = Polygon(
            [
                Point(self.center.x, self.center.y - self.ry),
                Point(self.center.x + self.rx, self.center.y),
                Point(self.center.x, self.center.y + self.ry),
                Point(self.center.x - self.rx, self.center.y),
            ]
        )

    def area(self) -> float:
        return 2.0 * self.rx * self.ry

    def perimeter(self) -> float:
        return 4.0 * math.hypot(self.rx, self.ry)

    def bbox(self) -> Rect:
        return Rect(
            self.center.x - self.rx,
            self.center.y - self.ry,
            self.center.x + self.rx,
            self.center.y + self.ry,
        )

    def centroid(self) -> Point:
        return Point(self.center.x, self.center.y)

    def sdf(self, pts: np.ndarray) -> np.ndarray:
        return self._poly.sdf(pts)


class Parallelogram(DiagramNode):
    """
    Skewed box defined by bounding center `center`, base `width`, `height`,
    and horizontal top/bottom skew offset `skew_x` (data I/O and stream nodes).
    """

    def __init__(
        self,
        center: Union[Point, Tuple[float, float]],
        width: float,
        height: float,
        skew_x: float = 12.0,
    ) -> None:
        self.center = center if isinstance(center, Point) else Point(center[0], center[1])
        self.w = float(width)
        self.h = float(height)
        self.skew_x = float(skew_x)
        cx, cy = self.center.x, self.center.y
        hw, hh = 0.5 * self.w, 0.5 * self.h
        s = 0.5 * self.skew_x
        self._poly = Polygon(
            [
                Point(cx - hw - s, cy - hh),
                Point(cx + hw - s, cy - hh),
                Point(cx + hw + s, cy + hh),
                Point(cx - hw + s, cy + hh),
            ]
        )

    def area(self) -> float:
        return self.w * self.h

    def perimeter(self) -> float:
        return self._poly.perimeter()

    def bbox(self) -> Rect:
        return self._poly.bbox()

    def centroid(self) -> Point:
        return Point(self.center.x, self.center.y)

    def sdf(self, pts: np.ndarray) -> np.ndarray:
        return self._poly.sdf(pts)


class RegularPolygon(DiagramNode):
    """Regular $N$-gon with circumradius `radius` centered at `center`."""

    def __init__(
        self,
        center: Union[Point, Tuple[float, float]],
        radius: float,
        num_sides: int = 6,
        angle_rad: float = 0.0,
    ) -> None:
        if num_sides < 3 or radius <= 0:
            raise ValueError("RegularPolygon requires num_sides >= 3 and radius > 0")
        self.center = center if isinstance(center, Point) else Point(center[0], center[1])
        self.radius = float(radius)
        self.num_sides = int(num_sides)
        self.angle_rad = float(angle_rad)

        verts = [
            Point(
                self.center.x
                + self.radius * math.cos(self.angle_rad + 2.0 * math.pi * i / self.num_sides),
                self.center.y
                + self.radius * math.sin(self.angle_rad + 2.0 * math.pi * i / self.num_sides),
            )
            for i in range(self.num_sides)
        ]
        self._poly = Polygon(verts)

    def area(self) -> float:
        n = self.num_sides
        return 0.5 * n * self.radius * self.radius * math.sin(2.0 * math.pi / n)

    def perimeter(self) -> float:
        n = self.num_sides
        return 2.0 * n * self.radius * math.sin(math.pi / n)

    def bbox(self) -> Rect:
        return self._poly.bbox()

    def centroid(self) -> Point:
        return Point(self.center.x, self.center.y)

    def sdf(self, pts: np.ndarray) -> np.ndarray:
        return self._poly.sdf(pts)


class Hexagon(RegularPolygon):
    """Regular 6-sided polygon (microservice / worker pod / hexagonal architecture node)."""

    def __init__(
        self,
        center: Union[Point, Tuple[float, float]],
        radius: float,
        flat_top: bool = True,
    ) -> None:
        ang = 0.0 if flat_top else math.pi / 6.0
        super().__init__(center, radius, num_sides=6, angle_rad=ang)


class Cylinder(DiagramNode):
    """
    2D orthographic projection of a 3D database cylinder with vertical body
    and top/bottom elliptical caps (`cap_ry`).
    """

    def __init__(
        self,
        center: Union[Point, Tuple[float, float]],
        width: float,
        height: float,
        cap_ry: float = None,
    ) -> None:
        self.center = center if isinstance(center, Point) else Point(center[0], center[1])
        self.w = float(width)
        self.h = float(height)
        rx = 0.5 * self.w
        ry = float(cap_ry) if cap_ry is not None else min(0.18 * self.h, 0.35 * rx)
        self.cap_ry = ry

        body_hh = max(1e-3, 0.5 * self.h - ry)
        cx, cy = self.center.x, self.center.y
        self.body = Rect(cx - rx, cy - body_hh, cx + rx, cy + body_hh)
        self.top_cap = Ellipse((cx, cy + body_hh), rx, ry)
        self.bottom_cap = Ellipse((cx, cy - body_hh), rx, ry)
        self._silhouette = self.body | self.top_cap | self.bottom_cap

    def area(self) -> float:
        return self.body.area() + self.top_cap.area()

    def perimeter(self) -> float:
        return 2.0 * self.body.height() + self.top_cap.perimeter()

    def bbox(self) -> Rect:
        cx, cy = self.center.x, self.center.y
        return Rect(cx - 0.5 * self.w, cy - 0.5 * self.h, cx + 0.5 * self.w, cy + 0.5 * self.h)

    def centroid(self) -> Point:
        return Point(self.center.x, self.center.y)

    def sdf(self, pts: np.ndarray) -> np.ndarray:
        return self._silhouette.sdf(pts)

    def top_rim_sdf(self, pts: np.ndarray) -> np.ndarray:
        """Signed distance to the top elliptical cap (useful for drawing the cylinder inner rim)."""
        return np.abs(self.top_cap.sdf(pts))


class CloudShape(DiagramNode):
    """
    Cloud / VPC boundary node built from the smooth union of overlapping circular lobes
    and a rounded base pill.
    """

    def __init__(
        self,
        center: Union[Point, Tuple[float, float]],
        width: float,
        height: float,
    ) -> None:
        self.center = center if isinstance(center, Point) else Point(center[0], center[1])
        self.w = float(width)
        self.h = float(height)
        cx, cy = self.center.x, self.center.y
        hw, hh = 0.5 * self.w, 0.5 * self.h

        self._parts: List[GeoShape] = [
            Capsule((cx - 0.52 * hw, cy - 0.28 * hh), (cx + 0.52 * hw, cy - 0.28 * hh), 0.42 * hh),
            Circle((cx - 0.42 * hw, cy + 0.02 * hh), 0.44 * hh),
            Circle((cx + 0.02 * hw, cy + 0.22 * hh), 0.52 * hh),
            Circle((cx + 0.44 * hw, cy - 0.02 * hh), 0.40 * hh),
        ]

    def area(self) -> float:
        return 0.72 * self.w * self.h

    def perimeter(self) -> float:
        return 2.4 * (self.w + self.h)

    def bbox(self) -> Rect:
        cx, cy = self.center.x, self.center.y
        return Rect(cx - 0.55 * self.w, cy - 0.45 * self.h, cx + 0.55 * self.w, cy + 0.55 * self.h)

    def centroid(self) -> Point:
        return Point(self.center.x, self.center.y)

    def sdf(self, pts: np.ndarray) -> np.ndarray:
        d = self._parts[0].sdf(pts)
        k = max(1e-3, 0.08 * min(self.w, self.h))
        for part in self._parts[1:]:
            dp = part.sdf(pts)
            h = np.clip(0.5 + 0.5 * (dp - d) / k, 0.0, 1.0)
            d = dp * (1.0 - h) + d * h - k * h * (1.0 - h)
        return d
