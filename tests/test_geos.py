import math
import unittest
import numpy as np

from geos import (
    Circle,
    Ellipse,
    Point,
    Polygon,
    Rect,
    RectSet,
    Segment,
    SmoothUnionShape,
    Triangle,
    anti_aliased_fill,
    make_grid,
    rasterize_mask,
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
        # Sum of disjoint piece areas must equal outer.area() - hole.area()
        self.assertAlmostEqual(
            sum(p.area() for p in pieces), outer.area() - hole.area()
        )
        # Pieces must be pairwise disjoint
        for i in range(len(pieces)):
            for j in range(i + 1, len(pieces)):
                self.assertEqual(pieces[i].intersection_area(pieces[j]), 0.0)

    def test_circle_and_ellipse_properties(self):
        c1 = Circle((0.0, 0.0), 5.0)
        c2 = Circle((0.0, 0.0), 3.0)
        # Concentric intersection area equals smaller circle area
        self.assertAlmostEqual(c1.intersection_area(c2), c2.area())
        # Disjoint circles have 0 intersection area
        c3 = Circle((20.0, 0.0), 2.0)
        self.assertEqual(c1.intersection_area(c3), 0.0)
        # Half-overlapping identical circles of radius R
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

        # Right triangle (3-4-5) has inradius = (a + b - c)/2 = 1.0 and circumradius = c/2 = 2.5
        inc = tri.incircle()
        circ = tri.circumcircle()
        self.assertAlmostEqual(inc.radius, 1.0)
        self.assertAlmostEqual(circ.radius, 2.5)

        u, v, w = tri.barycentric(tri.centroid())
        self.assertAlmostEqual(u, 1.0 / 3.0)
        self.assertAlmostEqual(v, 1.0 / 3.0)
        self.assertAlmostEqual(w, 1.0 / 3.0)

    def test_polygon_hull_clipping_and_triangulation(self):
        # L-shaped non-convex polygon (area = 10*10 - 6*6 = 64)
        l_poly = Polygon(
            [(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)]
        )
        self.assertAlmostEqual(l_poly.area(), 64.0)

        tris = l_poly.triangulate()
        self.assertEqual(len(tris), 4)
        self.assertAlmostEqual(sum(t.area() for t in tris), 64.0)

        # Convex hull of L-shape plus interior points has vertices (0,0),(10,0),(10,4),(4,10),(0,10) -> area = 82
        hull = Polygon.convex_hull(l_poly.vertices + [Point(2, 2), Point(3, 3)])
        self.assertAlmostEqual(hull.area(), 82.0)

        # Sutherland-Hodgman clipping against [0, 5] x [0, 5] square: area = 5*4 + 4*1 = 24
        clip_box = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        clipped_verts = l_poly.clip_with_convex(clip_box)
        clipped_poly = Polygon(clipped_verts)
        self.assertAlmostEqual(clipped_poly.area(), 24.0)


class TestRectSetAndCSGRaster(unittest.TestCase):
    def test_rectset_spatial_merger_and_sweep_line_disjoint_union(self):
        # 1. Transitive bounding merger + merge_allowed=False isolation
        rs = RectSet((500, 500), block_size=100, block_mode=True)
        rs.add_rect(Rect(10, 10, 40, 40))
        rs.add_rect(Rect(35, 20, 70, 50))
        rs.add_rect(Rect(20, 15, 60, 45, merge_allowed=False))
        self.assertEqual(len(rs), 2)

        # Spatial point and rectangle range queries
        hits_pt = rs.query_point((25, 25))
        self.assertEqual(len(hits_pt), 2)
        hits_box = rs.query_rect(Rect(65, 45, 80, 60))
        self.assertEqual(hits_box, [Rect(10, 10, 70, 50)])

        # 2. Exact sweep-line union area and disjoint decomposition (merge_overlaps=False)
        rs_exact = RectSet((500, 500), block_mode=False)
        r1 = Rect(0, 0, 10, 10)   # area 100
        r2 = Rect(5, 5, 15, 15)   # area 100, overlap with r1 is [5,10]x[5,10] = 25
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

        # Since c is completely inside r:
        # - r | c has the same mask as r
        # - r & c has the same mask as c
        # - (r - c) and (r & c) partition r into two disjoint sets
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

        # Smooth union SDF is bounded above by hard union SDF everywhere
        grid = make_grid(50, 50, bounds=(0, 0, 100, 100))
        self.assertTrue(np.all(smooth_shape.sdf(grid) <= union_shape.sdf(grid) + 1e-9))

        # Anti-aliased fill lies in [0, 1] and stroke mask extracts boundary band
        aa = anti_aliased_fill(c, 64, 64, bounds=(0, 0, 100, 100), edge_width=1.5)
        self.assertGreaterEqual(float(aa.min()), 0.0)
        self.assertLessEqual(float(aa.max()), 1.0)
        stroke = stroke_mask(c, stroke_width=3.0, height=64, width=64, bounds=(0, 0, 100, 100))
        self.assertTrue(np.any(stroke))
        self.assertFalse(stroke[32, 32])  # Center of circle is not on the boundary stroke


if __name__ == "__main__":
    unittest.main()
