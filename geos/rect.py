"""
Axis-aligned rectangle (`Rect`) supporting analytic CSG, sub-rectangle difference
decomposition, and vectorized NumPy SDF evaluation.
"""
from __future__ import annotations
import math
from typing import List, Optional, Tuple, Union
import numpy as np
from geos.interfaces import GeoShape
from geos.point import Point


class Rect(GeoShape):
    """
    Axis-aligned rectangle `[x1, x2] x [y1, y2]` with `x1 <= x2` and `y1 <= y2`.
    """

    def __init__(
        self,
        x1: float,
        y1: float,
        x2: float = 0.0,
        y2: float = 0.0,
        width: Optional[float] = None,
        height: Optional[float] = None,
        merge_allowed: bool = True,
    ) -> None:
        raw_x2 = x2 if width is None else x1 + width
        raw_y2 = y2 if height is None else y1 + height
        self.__x1: float = float(min(x1, raw_x2))
        self.__y1: float = float(min(y1, raw_y2))
        self.__x2: float = float(max(x1, raw_x2))
        self.__y2: float = float(max(y1, raw_y2))
        self.merge_allowed: bool = bool(merge_allowed)

    @property
    def x1(self) -> float:
        return self.__x1

    @property
    def y1(self) -> float:
        return self.__y1

    @property
    def x2(self) -> float:
        return self.__x2

    @property
    def y2(self) -> float:
        return self.__y2

    def width(self) -> float:
        return self.__x2 - self.__x1

    def height(self) -> float:
        return self.__y2 - self.__y1

    def area(self) -> float:
        return max(0.0, self.width()) * max(0.0, self.height())

    def perimeter(self) -> float:
        return 2.0 * (max(0.0, self.width()) + max(0.0, self.height()))

    def bbox(self) -> Rect:
        return Rect(
            self.__x1,
            self.__y1,
            self.__x2,
            self.__y2,
            merge_allowed=self.merge_allowed,
        )

    def centroid(self) -> Point:
        return Point(0.5 * (self.__x1 + self.__x2), 0.5 * (self.__y1 + self.__y2))

    def vertices(self) -> List[Point]:
        """Return the 4 counter-clockwise vertices starting at `(x1, y1)`."""
        return [
            Point(self.__x1, self.__y1),
            Point(self.__x2, self.__y1),
            Point(self.__x2, self.__y2),
            Point(self.__x1, self.__y2),
        ]

    def translate(self, dx: float, dy: float) -> Rect:
        return Rect(
            self.__x1 + dx,
            self.__y1 + dy,
            self.__x2 + dx,
            self.__y2 + dy,
            merge_allowed=self.merge_allowed,
        )

    def scale(
        self,
        sx: float,
        sy: Optional[float] = None,
        center: Optional[Point] = None,
    ) -> Rect:
        if sy is None:
            sy = sx
        c = center if center is not None else self.centroid()
        nx1 = c.x + (self.__x1 - c.x) * sx
        ny1 = c.y + (self.__y1 - c.y) * sy
        nx2 = c.x + (self.__x2 - c.x) * sx
        ny2 = c.y + (self.__y2 - c.y) * sy
        return Rect(nx1, ny1, nx2, ny2, merge_allowed=self.merge_allowed)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Rect):
            return False
        return (
            math.isclose(self.__x1, o.__x1, abs_tol=1e-9)
            and math.isclose(self.__x2, o.__x2, abs_tol=1e-9)
            and math.isclose(self.__y1, o.__y1, abs_tol=1e-9)
            and math.isclose(self.__y2, o.__y2, abs_tol=1e-9)
        )

    def __getitem__(self, index: Union[str, int]):
        if index in ("lowerLeft", "bottom_left", 0):
            return (self.__x1, self.__y1)
        if index in ("lowerRight", "bottom_right", 3):
            return (self.__x2, self.__y1)
        if index in ("upperRight", "top_right", 2):
            return (self.__x2, self.__y2)
        if index in ("upperLeft", "top_left", 1):
            return (self.__x1, self.__y2)
        if index == "width":
            return self.width()
        if index == "height":
            return self.height()
        if index == "area":
            return self.area()
        raise KeyError(f"Unknown Rect property: {index}")

    def contains_point(self, pt: Union[Point, Tuple[float, float]]) -> bool:
        """Return True if `(x, y)` lies inside or on the boundary of `self`."""
        px, py = float(pt[0]), float(pt[1])
        return (
            self.__x1 - 1e-9 <= px <= self.__x2 + 1e-9
            and self.__y1 - 1e-9 <= py <= self.__y2 + 1e-9
        )

    def is_overlapping(
        self, rect: Rect, padding: Union[int, float, Tuple[float, float]] = 0
    ) -> bool:
        """Check whether `self` and `rect` intersect when expanded by `padding`."""
        if isinstance(padding, (int, float)):
            pad_x, pad_y = float(padding), float(padding)
        else:
            pad_x, pad_y = float(padding[0]), float(padding[1])
        return not (
            self.__x1 > rect.__x2 + 2.0 * pad_x
            or self.__y1 > rect.__y2 + 2.0 * pad_y
            or self.__x2 < rect.__x1 - 2.0 * pad_x
            or self.__y2 < rect.__y1 - 2.0 * pad_y
        )

    isOverlapping = is_overlapping

    def intersection(self, rect: Rect) -> Optional[Rect]:
        """Return the exact intersection rectangle, or `None` if disjoint."""
        ix1 = max(self.__x1, rect.__x1)
        iy1 = max(self.__y1, rect.__y1)
        ix2 = min(self.__x2, rect.__x2)
        iy2 = min(self.__y2, rect.__y2)
        if ix2 <= ix1 or iy2 <= iy1:
            return None
        return Rect(ix1, iy1, ix2, iy2, merge_allowed=self.merge_allowed)

    def intersection_area(self, rect: Rect) -> float:
        inter = self.intersection(rect)
        return inter.area() if inter is not None else 0.0

    intersectionArea = intersection_area

    def bounding_union(self, rect: Rect) -> Rect:
        """Return the smallest axis-aligned rectangle enclosing both `self` and `rect`."""
        return Rect(
            min(self.__x1, rect.__x1),
            min(self.__y1, rect.__y1),
            max(self.__x2, rect.__x2),
            max(self.__y2, rect.__y2),
            merge_allowed=self.merge_allowed,
        )

    def iou(self, rect: Rect) -> float:
        """Intersection-over-Union (IoU) in `[0.0, 1.0]`."""
        inter = self.intersection_area(rect)
        if inter == 0.0:
            return 0.0
        union = self.area() + rect.area() - inter
        return inter / union if union > 0.0 else 0.0

    def update(
        self, rect: Rect, padding: float = 0.0, padded_update: bool = True
    ) -> None:
        """Expand `self` in-place to enclose `rect`."""
        pad = float(padding) if padded_update else 0.0
        self.__x1 = min(self.__x1, rect.__x1) - pad
        self.__y1 = min(self.__y1, rect.__y1) - pad
        self.__x2 = max(self.__x2, rect.__x2) + pad
        self.__y2 = max(self.__y2, rect.__y2) + pad

    def overlap_update(
        self, rect: Rect, padding: Union[int, float, Tuple[float, float]] = 0
    ) -> bool:
        """If `rect` overlaps `self`, merge `rect` into `self` and return True."""
        result = self.is_overlapping(rect, padding)
        if result:
            self.update(rect)
        return result

    overlapUpdate = overlap_update

    def get_blocks(self, block_size: int) -> List[Tuple[int, int]]:
        """Return `(row_block, col_block)` spatial grid cells covered by this rectangle."""
        x1 = max(0, int(self.__x1 // block_size))
        y1 = max(0, int(self.__y1 // block_size))
        x2 = max(x1 + 1, int(math.ceil(self.__x2 / block_size)))
        y2 = max(y1 + 1, int(math.ceil(self.__y2 / block_size)))
        return [(y, x) for y in range(y1, y2) for x in range(x1, x2)]

    getblocks = get_blocks

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"({self.__x1},{self.__y1})[]({self.__x2},{self.__y2})"
