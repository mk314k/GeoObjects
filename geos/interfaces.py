"""
Base interface (`GeoShape`) for 2D geometric shapes.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Union
import numpy as np

if TYPE_CHECKING:
    from geos.point import Point
    from geos.rect import Rect


class GeoShape:
    """
    Abstract base class for 2D geometric shapes supporting analytic queries,
    vectorized NumPy Signed Distance Fields (`sdf`), boolean raster masks (`to_mask`),
    and Constructive Solid Geometry operators (`|`, `&`, `-`, `^`).
    """

    def area(self) -> float:
        """Signed or unsigned 2D area of the shape."""
        raise NotImplementedError

    def perimeter(self) -> float:
        """Total boundary arc/edge length of the shape."""
        raise NotImplementedError

    def bbox(self) -> Rect:
        """Tight axis-aligned bounding box `(x1, y1, x2, y2)` enclosing the shape."""
        raise NotImplementedError

    def centroid(self) -> Point:
        """Geometric center of mass `(cx, cy)`."""
        raise NotImplementedError

    def contains(
        self, pts: Union[Point, Tuple[float, float], np.ndarray]
    ) -> Union[bool, np.ndarray]:
        """
        Return True (for a single 2D point) or a boolean NumPy array (for an `(..., 2)`
        coordinate array) indicating whether points lie inside or on the boundary.
        """
        if isinstance(pts, np.ndarray):
            return self.sdf(pts) <= 1e-9
        x, y = float(pts[0]), float(pts[1])
        d = self.sdf(np.array([[x, y]], dtype=np.float64))
        return bool(d.flat[0] <= 1e-9)

    def sdf(self, pts: np.ndarray) -> np.ndarray:
        """
        Evaluate the Signed Distance Field on an array `pts` of shape `(..., 2)`.
        Negative inside the shape, zero on the boundary, positive outside.
        """
        raise NotImplementedError

