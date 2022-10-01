# GeoObjects (`geos`)

2D geometric primitives, software architecture diagram nodes, $2 \times 2$ matrix & space transformations, spatial-grid rectangle merging, Constructive Solid Geometry (CSG), 2D convolutions, and SDF styling/effects.

## Overview

`geos` provides analytic and Signed Distance Field (SDF) representations for 2D shapes, diagramming components, affine and non-linear space transformations, multi-rectangle spatial indexing (`RectSet`), separable/2D convolutions, and multi-layer RGBA rendering (`Canvas`).

### Modules

- **`geos/interfaces.py`**: Base `GeoShape` interface with CSG operators (`|`, `&`, `-`, `^`), affine transformation helpers (`.translate()`, `.scale()`, `.rotate()`, `.shear()`, `.transform()`), and morphological SDF offsets (`.dilate()`, `.erode()`, `.shell()`).
- **`geos/point.py` & `geos/rect.py`**:
  - `Point` & `Segment`: 2D vector arithmetic, dot/cross products, segment-segment intersection, closest-point projection, and capsule SDF.
  - `Rect`: Axis-aligned rectangle with exact intersection, bounding union, `difference_rects` (decomposing $A \setminus B$ into up to 4 pairwise disjoint sub-rectangles), IoU, and box SDF.
- **`geos/circle.py`, `geos/triangle.py` & `geos/polygon.py`**:
  - `Circle` & `Ellipse`: Analytic circle-circle lens intersection area, Ramanujan perimeter, rotated bounding boxes, and Euclidean/implicit SDFs.
  - `Triangle`: Barycentric coordinates, incircle, circumcircle, and exact 2D triangle SDF.
  - `Polygon`: Shoelace signed area, area-weighted centroid, Monotone Chain $O(N \log N)$ convex hull, Sutherland-Hodgman convex clipping, ear-clipping triangulation, and ray-cast winding + segment SDF.
- **`geos/diagrams.py` (Software Architecture & Flowchart Nodes)**:
  - `DiagramNode` port anchors (`top`, `bottom`, `left`, `right`, `center`).
  - `RoundedRect`, `Capsule`, `Diamond`, `Parallelogram`, `RegularPolygon`, `Hexagon`, `Cylinder` (3D database projection), `CloudShape` (smooth-CSG lobes), `Arrow`, and `OrthogonalConnector` (Manhattan right-angle routing).
- **`geos/transforms.py` ($2 \times 2$ Matrix & Space Transformations)**:
  - `Transform2D`: Arbitrary $2 \times 2$ linear matrix $A$ + offset $t$ ($p' = A p + t$), supporting rotation, scaling, shear, reflection across an arbitrary axis, matrix composition (`T1 @ T2`), and exact inverse (`T.inverse()`).
  - `TransformedShape`: Evaluates any `GeoShape` under inverse-mapped coordinates with singular-value metric SDF scaling.
  - `GridRepeatShape`, `RadialRepeatShape`, and `TwistWarpShape` for $O(1)$ domain repetition and non-linear space warping.
- **`geos/rect_set.py` (Spatial Grid & Sweep-Line Union)**:
  - **2D Spatial-Block Transitive Merger (`add_rect`)**: Partitions space into `block_size x block_size` cells to merge overlapping rectangles into connected-component bounding boxes and accelerate `query_point` / `query_rect`.
  - **Sweep-Line Exact Union & Disjoint Decomposition (`union_area`, `decompose_disjoint`)**: Slices overlapping rectangles across $X$-events and merges 1D $Y$-intervals into pairwise disjoint rectangles.
- **`geos/convolution.py` & `geos/styling.py` (Filtering, Effects & RGBA Compositor)**:
  - `convolve1d_axis`, `convolve_separable`, `convolve2d`, `gaussian_blur`, `box_blur`, `sobel_gradients`, `sharpen`, `OffsetShape`, and `ShellShape`.
  - `SolidFill`, `LinearGradient`, `RadialGradient`, `StrokeStyle`, `DropShadow`, `OuterGlow`, `BevelEffect` (2.5D Sobel normal shading), and `Canvas` (Porter-Duff `alpha_over` RGBA compositor).

## Usage

```python
from geos import (
    BevelEffect,
    Canvas,
    Cylinder,
    DropShadow,
    LinearGradient,
    OrthogonalConnector,
    RoundedRect,
    StrokeStyle,
)

canvas = Canvas(height=240, width=420, bounds=(0, 0, 420, 240), background="#F8FAFC")

api_node = RoundedRect(40, 70, 170, 170, radius=12.0)
db_node = Cylinder((320, 120), width=90, height=110)
connector = OrthogonalConnector([api_node.port("right"), db_node.port("left")], shaft_width=4.0)

canvas.draw(
    api_node,
    fill=LinearGradient((40, 70), (170, 170), "#3B82F6", "#1D4ED8"),
    stroke=StrokeStyle(width=2.0, color="#1E293B"),
    shadow=DropShadow(offset=(4.0, -4.0), blur_sigma=4.0),
    bevel=BevelEffect(depth=5.0),
)
canvas.draw(db_node, fill="#10B981", stroke=StrokeStyle(width=2.0, color="#064E3B"))
canvas.draw(connector, fill="#334155")

rgba_image = canvas.to_uint8()
```

## Running Tests

```bash
python3 -m unittest discover -s tests -v
```

## License

MIT License
