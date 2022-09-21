"""
Base interface (`GeoShape`) for 2D geometric shapes.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Tuple, Union
import numpy as np

if TYPE_CHECKING:
    from geos.point import Point
    from geos.rect import Rect
    from geos.transforms import Transform2D


class GeoShape:
    """
    Abstract base class for 2D geometric shapes supporting analytic queries,
    vectorized NumPy Signed Distance Fields (`sdf`), boolean raster masks (`to_mask`),
    2x2 affine transformations, morphological offsets, and Constructive Solid Geometry
    operators (`|`, `&`, `-`, `^`).
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

    def to_mask(
        self,
        height: int,
        width: int,
        bounds: Tuple[float, float, float, float] = None,
    ) -> np.ndarray:
        """Rasterize the shape into a `(height, width)` boolean NumPy mask."""
        from geos.raster import rasterize_mask

        return rasterize_mask(self, height, width, bounds=bounds)

    def transform(self, xform: Transform2D) -> GeoShape:
        """Apply a `Transform2D` affine transformation to this shape."""
        from geos.transforms import TransformedShape

        return TransformedShape(self, xform)

    def translate(self, dx: float, dy: float) -> GeoShape:
        from geos.transforms import Transform2D, TransformedShape

        return TransformedShape(self, Transform2D.translation(dx, dy))

    def scale(
        self,
        sx: float,
        sy: Optional[float] = None,
        center: Optional[Union[Point, Tuple[float, float]]] = None,
    ) -> GeoShape:
        from geos.transforms import Transform2D, TransformedShape

        c = center if center is not None else self.centroid()
        return TransformedShape(self, Transform2D.scaling(sx, sy, center=c))

    def rotate(
        self,
        angle_rad: float,
        center: Optional[Union[Point, Tuple[float, float]]] = None,
    ) -> GeoShape:
        from geos.transforms import Transform2D, TransformedShape

        c = center if center is not None else self.centroid()
        return TransformedShape(self, Transform2D.rotation(angle_rad, center=c))

    def shear(
        self,
        shx: float = 0.0,
        shy: float = 0.0,
        center: Optional[Union[Point, Tuple[float, float]]] = None,
    ) -> GeoShape:
        from geos.transforms import Transform2D, TransformedShape

        c = center if center is not None else self.centroid()
        return TransformedShape(self, Transform2D.shear(shx, shy, center=c))

    def __or__(self, other: GeoShape) -> GeoShape:
        from geos.csg import UnionShape

        return UnionShape(self, other)

    def __and__(self, other: GeoShape) -> GeoShape:
        from geos.csg import IntersectionShape

        return IntersectionShape(self, other)

    def __sub__(self, other: GeoShape) -> GeoShape:
        from geos.csg import DifferenceShape

        return DifferenceShape(self, other)

    def __xor__(self, other: GeoShape) -> GeoShape:
        from geos.csg import SymmetricDifferenceShape

        return SymmetricDifferenceShape(self, other)
