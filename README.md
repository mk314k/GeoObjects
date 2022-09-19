# GeoObjects (`geos`)

2D geometric primitives, spatial-grid rectangle merging, Constructive Solid Geometry (CSG), and vectorized NumPy Signed Distance Fields (SDFs).

## Overview

`geos` provides analytic 2D geometric shapes (`Point`, `Segment`, `Rect`, `Circle`, `Ellipse`, `Triangle`, `Polygon`), multi-rectangle spatial indexing and exact union decomposition (`RectSet`), boolean CSG expression trees (`|`, `&`, `-`, `^`, `SmoothUnionShape`), and vectorized NumPy rasterization (`sdf`, `to_mask`, `anti_aliased_fill`, `stroke_mask`).

### Modules

- `geos/interfaces.py`: Base `GeoShape` interface with CSG operator overloads (`|`, `&`, `-`, `^`).
- `geos/point.py`: `Point` (2D vector arithmetic, dot/cross product, rotation, linear interpolation) and `Segment` (segment-segment intersection, closest-point projection, capsule SDF).
- `geos/rect.py`: `Rect` (axis-aligned bounding box, exact intersection, bounding union, `difference_rects` decomposing $A \setminus B$ into up to 4 disjoint sub-rectangles, IoU, and box SDF).
- `geos/circle.py`: `Circle` (analytic lens intersection area, circle-rect overlap, Euclidean SDF) and `Ellipse` (Ramanujan perimeter, rotated bounding box, gradient-normalized implicit SDF).
- `geos/triangle.py`: `Triangle` (signed area, barycentric coordinates, incircle, circumcircle, exact triangle SDF).
- `geos/polygon.py`: `Polygon` (shoelace area, area-weighted centroid, Monotone Chain $O(N \log N)$ convex hull, Sutherland-Hodgman convex clipping, ear-clipping triangulation, and ray-cast winding + segment SDF).
- `geos/rect_set.py`:
  - **2D Spatial-Block Transitive Merger (`add_rect`)**: Partitions space into `block_size x block_size` cells to merge overlapping rectangles into connected-component bounding boxes.
  - **Sweep-Line Exact Union & Disjoint Decomposition (`union_area`, `decompose_disjoint`)**: Slices overlapping rectangles along $X$-events and merges 1D $Y$-intervals to compute exact union area and partition the union into pairwise disjoint rectangles.
- `geos/csg.py` & `geos/raster.py`: Boolean CSG nodes (`UnionShape`, `IntersectionShape`, `DifferenceShape`, `SymmetricDifferenceShape`, `SmoothUnionShape`) and vectorized NumPy grid evaluators (`make_grid`, `render_sdf`, `rasterize_mask`, `anti_aliased_fill`, `stroke_mask`).

## Usage

```python
from geos import Circle, Polygon, Rect, RectSet, anti_aliased_fill

# 1. Boolean CSG and vectorized NumPy anti-aliased rasterization
plate = Rect(10, 10, 90, 90) - Circle((50, 50), radius=22)
coverage = anti_aliased_fill(plate, height=256, width=256, bounds=(0, 0, 100, 100))

# 2. Exact sweep-line union area and disjoint decomposition of overlapping rectangles
rset = RectSet((500, 500), block_mode=False)
rset.add_rect(Rect(0, 0, 40, 40), merge_overlaps=False)
rset.add_rect(Rect(20, 20, 60, 60), merge_overlaps=False)
print("Exact union area:", rset.union_area())
print("Disjoint rectangles:", rset.decompose_disjoint())
```

## Running Tests

```bash
python3 -m unittest discover -s tests -v
```

## License

MIT License
