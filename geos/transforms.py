"""
2x2 linear matrix + translation transformations (`Transform2D`), inverse-mapped
shape wrappers (`TransformedShape`), and spatial domain warps (`GridRepeatShape`,
`RadialRepeatShape`, `TwistWarpShape`).
"""
from __future__ import annotations
import math
from typing import Optional, Sequence, Tuple, Union
import numpy as np
from geos.interfaces import GeoShape
from geos.point import Point
from geos.rect import Rect


class Transform2D:
    """
    2D affine transform `p' = A @ p + t`, where `A` is a 2x2 matrix and `t` is a
    2D translation vector.
    """

    def __init__(
        self,
        matrix: Optional[Union[Sequence[Sequence[float]], np.ndarray]] = None,
        offset: Union[Point, Tuple[float, float], np.ndarray] = (0.0, 0.0),
    ) -> None:
        if matrix is None:
            self.matrix = np.eye(2, dtype=np.float64)
        else:
            arr = np.asarray(matrix, dtype=np.float64)
            if arr.shape != (2, 2):
                raise ValueError(f"Transform2D matrix must be (2, 2), got {arr.shape}")
            self.matrix = arr.copy()
        self.offset = np.array([float(offset[0]), float(offset[1])], dtype=np.float64)

    @staticmethod
    def identity() -> Transform2D:
        return Transform2D()

    @staticmethod
    def translation(dx: float, dy: float) -> Transform2D:
        return Transform2D(np.eye(2, dtype=np.float64), (dx, dy))

    @staticmethod
    def scaling(
        sx: float,
        sy: Optional[float] = None,
        center: Union[Point, Tuple[float, float]] = (0.0, 0.0),
    ) -> Transform2D:
        if sy is None:
            sy = sx
        m = np.array([[float(sx), 0.0], [0.0, float(sy)]], dtype=np.float64)
        c = np.array([float(center[0]), float(center[1])], dtype=np.float64)
        t = c - m @ c
        return Transform2D(m, (t[0], t[1]))

    @staticmethod
    def rotation(
        angle_rad: float,
        center: Union[Point, Tuple[float, float]] = (0.0, 0.0),
    ) -> Transform2D:
        cs, sn = math.cos(angle_rad), math.sin(angle_rad)
        m = np.array([[cs, -sn], [sn, cs]], dtype=np.float64)
        c = np.array([float(center[0]), float(center[1])], dtype=np.float64)
        t = c - m @ c
        return Transform2D(m, (t[0], t[1]))

    @staticmethod
    def shear(
        shx: float = 0.0,
        shy: float = 0.0,
        center: Union[Point, Tuple[float, float]] = (0.0, 0.0),
    ) -> Transform2D:
        m = np.array([[1.0, float(shx)], [float(shy), 1.0]], dtype=np.float64)
        c = np.array([float(center[0]), float(center[1])], dtype=np.float64)
        t = c - m @ c
        return Transform2D(m, (t[0], t[1]))

    @staticmethod
    def reflection(
        axis: Union[Point, Tuple[float, float]] = (1.0, 0.0),
        center: Union[Point, Tuple[float, float]] = (0.0, 0.0),
    ) -> Transform2D:
        """Reflect across a line through `center` in direction `axis = (ux, uy)`."""
        ux, uy = float(axis[0]), float(axis[1])
        norm = math.hypot(ux, uy)
        if norm <= 1e-15:
            raise ValueError("Reflection axis must be non-zero")
        ux /= norm
        uy /= norm
        m = np.array(
            [
                [2.0 * ux * ux - 1.0, 2.0 * ux * uy],
                [2.0 * ux * uy, 2.0 * uy * uy - 1.0],
            ],
            dtype=np.float64,
        )
        c = np.array([float(center[0]), float(center[1])], dtype=np.float64)
        t = c - m @ c
        return Transform2D(m, (t[0], t[1]))

    def determinant(self) -> float:
        return float(
            self.matrix[0, 0] * self.matrix[1, 1]
            - self.matrix[0, 1] * self.matrix[1, 0]
        )

    def min_singular_value(self) -> float:
        """Smallest singular value `sigma_min(A)`, used for metric SDF scaling."""
        sv = np.linalg.svd(self.matrix, compute_uv=False)
        return float(sv[-1])

    def inverse(self) -> Transform2D:
        det = self.determinant()
        if abs(det) <= 1e-15:
            raise np.linalg.LinAlgError("Transform2D matrix is singular")
        inv_m = np.linalg.inv(self.matrix)
        inv_t = -(inv_m @ self.offset)
        return Transform2D(inv_m, (inv_t[0], inv_t[1]))

    def apply_point(self, pt: Union[Point, Tuple[float, float]]) -> Point:
        v = np.array([float(pt[0]), float(pt[1])], dtype=np.float64)
        res = self.matrix @ v + self.offset
        return Point(res[0], res[1])

    def apply(self, pts: np.ndarray) -> np.ndarray:
        """Apply transform to an array `pts` of shape `(..., 2)`."""
        pts_arr = np.asarray(pts, dtype=np.float64)
        return pts_arr @ self.matrix.T + self.offset

    def __matmul__(self, other: Transform2D) -> Transform2D:
        """Compose transforms `(self @ other)(p) = self(other(p))`."""
        m = self.matrix @ other.matrix
        t = self.matrix @ other.offset + self.offset
        return Transform2D(m, (t[0], t[1]))

    def __repr__(self) -> str:
        return f"Transform2D(matrix={self.matrix.tolist()}, offset={self.offset.tolist()})"
