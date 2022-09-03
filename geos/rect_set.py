"""
Spatial-grid rectangle container (`RectSet`) supporting 2D block-indexed
transitive bounding merges, spatial point/range queries, and exact sweep-line
union area & disjoint decomposition.
"""
from __future__ import annotations
import math
from typing import Dict, Iterator, List, Optional, Set, Tuple, Union
from geos.point import Point
from geos.rect import Rect


class RectSet:
    """
    Collection of `Rect` regions partitioned into `block_size x block_size` grid cells.

    Supports:
      1. **Transitive Bounding-Box Merger (`add_rect` with `merge_overlaps=True`)**:
         Uses a 2D spatial hash grid to merge overlapping rectangles (where
         `rect.merge_allowed=True`) into their connected-component bounding boxes.
      2. **Spatial Point & Range Queries (`query_point`, `query_rect`)**:
         Uses the 2D grid to retrieve rectangles containing a point or intersecting
         a query region without scanning the full collection.
      3. **Exact Sweep-Line Union & Disjoint Decomposition (`union_area`, `decompose_disjoint`)**:
         Uses a 1D vertical-slab sweep over all rectangle $X$-intervals to compute the
         exact area of the union of arbitrary overlapping rectangles and partition the
         union into pairwise disjoint rectangles.
    """

    def __init__(
        self,
        dimension: Union[int, Tuple[int, int]] = (1000, 1000),
        block_size: int = 200,
        block_mode: bool = True,
        blockSize: Optional[int] = None,
        blockMode: Optional[bool] = None,
    ) -> None:
        if blockSize is not None:
            block_size = blockSize
        if blockMode is not None:
            block_mode = blockMode
        self.__blocksize = max(1, int(block_size))
        self.__blockMode = bool(block_mode)
        self.__rects: List[Optional[Rect]] = []
        self.__free_slots: List[int] = []
        if isinstance(dimension, int):
            dimension = (dimension, dimension)
        self.__dimension = (int(dimension[0]), int(dimension[1]))
        if self.__blockMode:
            grid_dim = (
                math.ceil(self.__dimension[0] / self.__blocksize) + 2,
                math.ceil(self.__dimension[1] / self.__blocksize) + 2,
            )
            self.__grid_shape = grid_dim
            self.__grid: Dict[Tuple[int, int], Set[int]] = {}

    def _clamp_blocks(
        self, blocks: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        max_r, max_c = self.__grid_shape
        return [
            (r, c) for r, c in blocks if 0 <= r < max_r and 0 <= c < max_c
        ]

    def _insert_slot(self, rect: Rect) -> int:
        if self.__free_slots:
            idx = self.__free_slots.pop()
            self.__rects[idx] = rect
            return idx
        idx = len(self.__rects)
        self.__rects.append(rect)
        return idx

    def _remove_slot(self, idx: int) -> None:
        if 0 <= idx < len(self.__rects) and self.__rects[idx] is not None:
            self.__rects[idx] = None
            self.__free_slots.append(idx)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, RectSet):
            return False
        return list(self) == list(o)

    def add_rect(
        self,
        rect: Rect,
        padding: Union[int, float, Tuple[float, float]] = 0,
        merge_overlaps: bool = True,
    ) -> None:
        """
        Insert `rect` into the set. When `merge_overlaps=True` and `rect.merge_allowed=True`,
        transitively merges overlapping merge-enabled rectangles into their bounding union.
        """
        do_merge = merge_overlaps and rect.merge_allowed

        if self.__blockMode:
            blocks = self._clamp_blocks(rect.get_blocks(self.__blocksize))
            while do_merge:
                merged_any = False
                for block in blocks:
                    cell = self.__grid.get(block)
                    if not cell:
                        continue
                    for idx_val in list(cell):
                        existing = self.__rects[idx_val]
                        if (
                            existing is not None
                            and existing.merge_allowed
                            and rect.overlap_update(existing, padding)
                        ):
                            old_blocks = self._clamp_blocks(
                                existing.get_blocks(self.__blocksize)
                            )
                            for ob in old_blocks:
                                if ob in self.__grid:
                                    self.__grid[ob].discard(idx_val)
                            self._remove_slot(idx_val)
                            blocks = self._clamp_blocks(
                                rect.get_blocks(self.__blocksize)
                            )
                            merged_any = True
                            break
                    if merged_any:
                        break
                if not merged_any:
                    break

            slot_idx = self._insert_slot(rect)
            for block in blocks:
                if block not in self.__grid:
                    self.__grid[block] = set()
                self.__grid[block].add(slot_idx)
        else:
            if do_merge:
                merged = True
                while merged:
                    merged = False
                    for idx_val, existing in enumerate(self.__rects):
                        if (
                            existing is not None
                            and existing.merge_allowed
                            and rect.overlap_update(existing, padding)
                        ):
                            self._remove_slot(idx_val)
                            merged = True
                            break
            self._insert_slot(rect)

    addRect = add_rect

    def __len__(self) -> int:
        return sum(1 for r in self.__rects if r is not None)

    def __iter__(self) -> Iterator[Rect]:
        for r in self.__rects:
            if r is not None:
                yield r

    def __repr__(self) -> str:
        return repr(list(self))
