import math
import unittest
import numpy as np

from geos import (
    Arrow,
    BevelEffect,
    Canvas,
    Capsule,
    Circle,
    CloudShape,
    Cylinder,
    Diamond,
    DropShadow,
    Ellipse,
    GridRepeatShape,
    Hexagon,
    LinearGradient,
    OrthogonalConnector,
    OuterGlow,
    Parallelogram,
    Point,
    Polygon,
    RadialGradient,
    RadialRepeatShape,
    Rect,
    RectSet,
    RoundedRect,
    Segment,
    SmoothUnionShape,
    StrokeStyle,
    Transform2D,
    Triangle,
    TwistWarpShape,
    anti_aliased_fill,
    convolve2d,
    convolve_separable,
    gaussian_blur,
    gaussian_kernel_1d,
    make_grid,
    rasterize_mask,
    sobel_gradients,
    stroke_mask,
)


class TestPrimitivesAndAnalyticGeometry(unittest.TestCase):
    def test_point_and_segment_geometry(self):
        p1 = Point(0.0, 0.0)
        p2 = Point(4.0, 0.0)
        p_rot = p2.rotate(math.pi / 2.0)
        self.assertEqual(p_rot, Point(0.0, 4.0))

        seg1 = Segment((0.0, 0.0), (4.0, 4.0))
        seg2 = Segment((0.0, 4.0), (4.0, 0.0))
        inter = seg1.intersection(seg2)
        self.assertIsNotNone(inter)
        self.assertEqual(inter, Point(2.0, 2.0))

        seg_h = Segment((0.0, 0.0), (10.0, 0.0))
        self.assertEqual(seg_h.closest_point((5.0, 3.0)), Point(5.0, 0.0))
        self.assertAlmostEqual(seg_h.distance_to_point((5.0, 3.0)), 3.0)

    def test_rect_analytic_ops_and_difference_decomposition(self):
        outer = Rect(0.0, 0.0, 10.0, 10.0)
        hole = Rect(3.0, 3.0, 7.0, 7.0)
        pieces = outer.difference_rects(hole)
        self.assertEqual(len(pieces), 4)
        self.assertAlmostEqual(
            sum(p.area() for p in pieces), outer.area() - hole.area()
        )
        for i in range(len(pieces)):
            for j in range(i + 1, len(pieces)):
                self.assertEqual(pieces[i].intersection_area(pieces[j]), 0.0)

    def test_circle_and_ellipse_properties(self):
        c1 = Circle((0.0, 0.0), 5.0)
        c2 = Circle((0.0, 0.0), 3.0)
        self.assertAlmostEqual(c1.intersection_area(c2), c2.area())
        c3 = Circle((20.0, 0.0), 2.0)
        self.assertEqual(c1.intersection_area(c3), 0.0)
        c_shift = Circle((5.0, 0.0), 5.0)
        expected = 25.0 * (2.0 * math.pi / 3.0 - math.sqrt(3.0) / 2.0)
        self.assertAlmostEqual(c1.intersection_area(c_shift), expected, places=6)

        el = Ellipse((10.0, 20.0), rx=6.0, ry=3.0)
        self.assertAlmostEqual(el.area(), math.pi * 18.0)
        self.assertTrue(el.contains((10.0, 20.0)))
        self.assertFalse(el.contains((20.0, 20.0)))

    def test_triangle_barycentric_circumcircle_and_incircle(self):
        tri = Triangle((0.0, 0.0), (4.0, 0.0), (0.0, 3.0))
        self.assertAlmostEqual(tri.area(), 6.0)
        self.assertAlmostEqual(tri.perimeter(), 12.0)

        inc = tri.incircle()
        circ = tri.circumcircle()
        self.assertAlmostEqual(inc.radius, 1.0)
        self.assertAlmostEqual(circ.radius, 2.5)

        u, v, w = tri.barycentric(tri.centroid())
        self.assertAlmostEqual(u, 1.0 / 3.0)
        self.assertAlmostEqual(v, 1.0 / 3.0)
        self.assertAlmostEqual(w, 1.0 / 3.0)

    def test_polygon_hull_clipping_and_triangulation(self):
        l_poly = Polygon(
            [(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)]
        )
        self.assertAlmostEqual(l_poly.area(), 64.0)

        tris = l_poly.triangulate()
        self.assertEqual(len(tris), 4)
        self.assertAlmostEqual(sum(t.area() for t in tris), 64.0)

        hull = Polygon.convex_hull(l_poly.vertices + [Point(2, 2), Point(3, 3)])
        self.assertAlmostEqual(hull.area(), 82.0)

        clip_box = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        clipped_verts = l_poly.clip_with_convex(clip_box)
        clipped_poly = Polygon(clipped_verts)
        self.assertAlmostEqual(clipped_poly.area(), 24.0)


class TestRectSetAndCSGRaster(unittest.TestCase):
    def test_rectset_spatial_merger_and_sweep_line_disjoint_union(self):
        rs = RectSet((500, 500), block_size=100, block_mode=True)
        rs.add_rect(Rect(10, 10, 40, 40))
        rs.add_rect(Rect(35, 20, 70, 50))
        rs.add_rect(Rect(20, 15, 60, 45, merge_allowed=False))
        self.assertEqual(len(rs), 2)

        hits_pt = rs.query_point((25, 25))
        self.assertEqual(len(hits_pt), 2)
        hits_box = rs.query_rect(Rect(65, 45, 80, 60))
        self.assertEqual(hits_box, [Rect(10, 10, 70, 50)])

        rs_exact = RectSet((500, 500), block_mode=False)
        r1 = Rect(0, 0, 10, 10)
        r2 = Rect(5, 5, 15, 15)
        rs_exact.add_rect(r1, merge_overlaps=False)
        rs_exact.add_rect(r2, merge_overlaps=False)
        self.assertAlmostEqual(rs_exact.union_area(), 175.0)

        disjoint = rs_exact.decompose_disjoint()
        self.assertAlmostEqual(sum(d.area() for d in disjoint), 175.0)
        for i in range(len(disjoint)):
            for j in range(i + 1, len(disjoint)):
                self.assertEqual(disjoint[i].intersection_area(disjoint[j]), 0.0)

    def test_csg_boolean_operators_and_vectorized_sdf(self):
        r = Rect(20.0, 20.0, 80.0, 80.0)
        c = Circle((50.0, 50.0), 20.0)

        union_shape = r | c
        inter_shape = r & c
        diff_shape = r - c
        xor_shape = r ^ c
        smooth_shape = SmoothUnionShape(r, c, k=4.0)

        mask_r = rasterize_mask(r, 100, 100, bounds=(0, 0, 100, 100))
        mask_c = rasterize_mask(c, 100, 100, bounds=(0, 0, 100, 100))
        mask_u = rasterize_mask(union_shape, 100, 100, bounds=(0, 0, 100, 100))
        mask_i = rasterize_mask(inter_shape, 100, 100, bounds=(0, 0, 100, 100))
        mask_d = rasterize_mask(diff_shape, 100, 100, bounds=(0, 0, 100, 100))
        mask_x = rasterize_mask(xor_shape, 100, 100, bounds=(0, 0, 100, 100))

        np.testing.assert_array_equal(mask_u, mask_r)
        np.testing.assert_array_equal(mask_i, mask_c)
        np.testing.assert_array_equal(mask_d, mask_x)
        np.testing.assert_array_equal(mask_d | mask_i, mask_r)
        self.assertFalse(np.any(mask_d & mask_i))

        grid = make_grid(50, 50, bounds=(0, 0, 100, 100))
        self.assertTrue(np.all(smooth_shape.sdf(grid) <= union_shape.sdf(grid) + 1e-9))

        aa = anti_aliased_fill(c, 64, 64, bounds=(0, 0, 100, 100), edge_width=1.5)
        self.assertGreaterEqual(float(aa.min()), 0.0)
        self.assertLessEqual(float(aa.max()), 1.0)
        stroke = stroke_mask(c, stroke_width=3.0, height=64, width=64, bounds=(0, 0, 100, 100))
        self.assertTrue(np.any(stroke))
        self.assertFalse(stroke[32, 32])


class TestDiagramsTransformsConvolutionAndStyling(unittest.TestCase):
    def test_architecture_diagram_shapes_and_ports(self):
        service = RoundedRect(20, 40, 120, 100, radius=10.0)
        db = Cylinder((220, 70), width=70, height=80)
        decision = Diamond((150, 180), rx=35, ry=25)
        io_node = Parallelogram((300, 180), width=80, height=40, skew_x=14)
        pod = Hexagon((320, 70), radius=30)
        cloud = CloudShape((170, 260), width=120, height=70)
        queue = Capsule((40, 240), (100, 240), radius=15)

        for node in (service, db, decision, io_node, pod, cloud, queue):
            c = node.centroid()
            self.assertTrue(node.contains(c))
            self.assertGreater(node.area(), 0.0)
            ports = node.ports()
            self.assertEqual(set(ports.keys()), {"center", "top", "bottom", "left", "right"})

        conn = OrthogonalConnector([service.port("right"), db.port("left")], shaft_width=4.0)
        arr = Arrow((20, 20), (80, 20), shaft_width=4.0, head_length=12.0, head_width=12.0)
        self.assertTrue(arr.contains((50, 20)))
        self.assertGreater(conn.area(), 0.0)

    def test_matrix_2x2_transforms_and_space_warps(self):
        # Composition with inverse must yield identity
        t_rot = Transform2D.rotation(math.pi / 3.0, center=(15.0, -10.0))
        t_shear = Transform2D.shear(0.4, -0.2, center=(5.0, 5.0))
        t_scale = Transform2D.scaling(2.0, 0.5)
        t_comp = t_rot @ t_shear @ t_scale
        t_id = t_comp @ t_comp.inverse()
        np.testing.assert_allclose(t_id.matrix, np.eye(2), atol=1e-12)
        np.testing.assert_allclose(t_id.offset, np.zeros(2), atol=1e-12)

        # Area under 2x2 linear transformation scales by |det(A)|
        base_rect = Rect(0.0, 0.0, 10.0, 20.0)
        warped = base_rect.transform(t_comp)
        self.assertAlmostEqual(warped.area(), base_rect.area() * abs(t_comp.determinant()))

        # Reflection across X-axis maps (3, 5) -> (3, -5)
        refl = Transform2D.reflection((1.0, 0.0))
        self.assertEqual(refl.apply_point((3.0, 5.0)), Point(3.0, -5.0))

        # Domain repetition and twist warps evaluate cleanly on grids
        dot = Circle((0.0, 0.0), 3.0)
        tiled = GridRepeatShape(dot, period_x=10.0, period_y=10.0, bounds=Rect(-25, -25, 25, 25))
        self.assertTrue(tiled.contains((0.0, 0.0)))
        self.assertTrue(tiled.contains((10.0, -20.0)))
        self.assertFalse(tiled.contains((5.0, 5.0)))

        spokes = RadialRepeatShape(Rect(8.0, -1.5, 20.0, 1.5), count=6, center=(0.0, 0.0))
        self.assertTrue(spokes.contains((14.0, 0.0)))
        self.assertTrue(spokes.contains((-14.0, 0.0)))

        twisted = TwistWarpShape(Rect(-10, -2, 10, 2), strength=0.05)
        self.assertTrue(twisted.contains((0.0, 0.0)))

    def test_convolutions_sobel_and_morphological_offsets(self):
        rng = np.random.default_rng(12)
        img = rng.random((32, 36))
        k1d = gaussian_kernel_1d(1.2)
        k2d = np.outer(k1d, k1d)

        # Separable 1D-1D convolution must match full 2D outer-product convolution
        sep_out = convolve_separable(img, k1d, k1d, pad_mode="edge")
        full_out = convolve2d(img, k2d, pad_mode="edge")
        np.testing.assert_allclose(sep_out, full_out, atol=1e-12)

        # Gaussian blur preserves constant fields
        const_field = np.full((20, 20), 7.5)
        np.testing.assert_allclose(gaussian_blur(const_field, sigma=2.0), const_field, atol=1e-12)

        # Sobel gradients on f(x, y) = 3*x - 2*y equal (3, -2) in the interior
        grid = make_grid(20, 20, bounds=(0.0, 0.0, 20.0, 20.0))
        linear_field = 3.0 * grid[..., 0] - 2.0 * grid[..., 1]
        gx, gy = sobel_gradients(linear_field)
        np.testing.assert_allclose(gx[2:-2, 2:-2], 3.0, atol=1e-9)
        np.testing.assert_allclose(gy[2:-2, 2:-2], -2.0, atol=1e-9)

        # Morphological SDF dilation/erosion/shell
        c = Circle((0.0, 0.0), 10.0)
        c_dil = c.dilate(3.0)
        c_ero = c.erode(3.0)
        c_sh = c.shell(2.0)
        self.assertTrue(c_dil.contains((12.5, 0.0)))
        self.assertFalse(c_ero.contains((8.0, 0.0)))
        self.assertTrue(c_sh.contains((10.0, 0.0)))
        self.assertFalse(c_sh.contains((0.0, 0.0)))

    def test_styling_effects_and_canvas_compositing(self):
        canvas = Canvas(64, 80, bounds=(0, 0, 80, 64), background="#F8FAFC")
        card = RoundedRect(12, 12, 68, 52, radius=6.0)
        canvas.draw(
            card,
            fill=LinearGradient((12, 12), (68, 52), "#3B82F6", "#1D4ED8"),
            stroke=StrokeStyle(width=2.0, color="#1E293B", align="center"),
            shadow=DropShadow(offset=(2.0, -2.0), blur_sigma=2.0, color=(0, 0, 0, 0.3)),
            glow=OuterGlow(radius=4.0, color=(0.2, 0.5, 1.0, 0.4)),
            bevel=BevelEffect(depth=4.0),
        )
        badge = Circle((40, 32), radius=10.0)
        canvas.draw(
            badge,
            fill=RadialGradient((40, 32), 10.0, "#FBBF24", "#D97706"),
        )
        rgba_u8 = canvas.to_uint8()
        self.assertEqual(rgba_u8.shape, (64, 80, 4))
        self.assertEqual(rgba_u8.dtype, np.uint8)
        # Center of badge should have high red/green (amber) channel intensity
        self.assertGreater(int(rgba_u8[32, 40, 0]), 180)


if __name__ == "__main__":
    unittest.main()
