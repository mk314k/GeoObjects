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



if __name__ == "__main__":
    unittest.main()
