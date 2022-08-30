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


    def isUpperRightTo(self, point: Point) -> bool:
        return self.__x >= point.__x and self.__y >= point.__y

    def minXYCompare(self, point: Point) -> Point:
        return Point(min(self.__x, point.__x), min(self.__y, point.__y))

    def maxXYCompare(self, point: Point) -> Point:
        return Point(max(self.__x, point.__x), max(self.__y, point.__y))
