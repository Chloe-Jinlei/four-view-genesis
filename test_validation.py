"""Small analytic cases test actual aperture crossing, rather than proximity."""

import json
import unittest

import numpy as np

try:
    from .validation import analyze_cloth, extract_boundary_loops
except ImportError:
    from validation import analyze_cloth, extract_boundary_loops


def cloth_with_two_holes():
    xs, ys = np.linspace(-2, 2, 9), np.linspace(-1, 1, 5)
    vertices = np.array([(x, y, 0.0) for y in ys for x in xs])
    triangles = []
    for row in range(len(ys) - 1):
        for column in range(len(xs) - 1):
            if row == 1 and column in (2, 5):
                continue
            a = row * len(xs) + column
            b, c, d = a + 1, a + len(xs), a + len(xs) + 1
            triangles.extend(((a, b, d), (a, d, c)))
    return vertices, np.asarray(triangles)


class ClothGeometryTest(unittest.TestCase):
    def setUp(self):
        self.rest, self.faces = cloth_with_two_holes()
        self.arms = {
            "left": ((-.75, -.25, .3), (-.75, -.25, -.3)),
            "right": ((.75, -.25, .3), (.75, -.25, -.3)),
        }

    def analyze(self, current=None):
        return analyze_cloth(self.rest, self.faces,
                             self.rest if current is None else current,
                             self.arms, (0, 0, -10), (.2, .2, .2))

    def test_two_true_crossings_and_json(self):
        self.assertEqual(sorted(map(len, extract_boundary_loops(self.faces))), [4, 4, 24])
        result = self.analyze()
        self.assertTrue(result["threading_passed"])
        left = result["sleeves"]["left"]["arm_crossings"]["left"]
        self.assertAlmostEqual(left["intersection_fraction_shoulder_to_tip"], .5)
        self.assertLess(left["shoulder_signed_distance_m"] * left["tip_signed_distance_m"], 0)
        json.dumps(result, allow_nan=False)

    def test_same_side_is_not_threaded_even_above_hole_center(self):
        self.arms["left"] = ((-.75, -.25, .3), (-.75, -.25, .02))
        result = self.analyze()
        self.assertFalse(result["threading_passed"])
        self.assertFalse(result["sleeves"]["left"]["arm_crossings"]["left"]["endpoints_straddle_plane"])

    def test_crossing_plane_outside_aperture_is_not_threaded(self):
        self.arms["left"] = ((-1.2, -.25, .3), (-1.2, -.25, -.3))
        result = self.analyze()
        self.assertFalse(result["threading_passed"])
        self.assertTrue(result["sleeves"]["left"]["arm_crossings"]["left"]["endpoints_straddle_plane"])

    def test_arm_on_aperture_boundary_fails_closed(self):
        self.arms["left"] = ((-1, -.25, .3), (-1, -.25, -.3))
        result = self.analyze()
        self.assertFalse(result["threading_passed"])
        self.assertTrue(result["sleeves"]["left"]["arm_crossings"]["left"]["intersection_on_boundary_ambiguous"])

    def test_nonplanar_aperture_is_unreliable(self):
        current = self.rest.copy()
        current[11, 2] = .7
        result = self.analyze(current)
        self.assertFalse(result["threading_passed"])
        self.assertFalse(result["sleeves"]["left"]["reliable"])

    def test_swapped_arms_do_not_pass_matching_requirement(self):
        self.arms["left"], self.arms["right"] = self.arms["right"], self.arms["left"]
        self.assertFalse(self.analyze()["threading_passed"])

    def test_uniform_stretch_is_measured(self):
        result = self.analyze(self.rest * 1.2)
        for value in result["edge_stretch"]["length_ratio"].values():
            self.assertAlmostEqual(value, 1.2)

    def test_nonmanifold_mesh_is_rejected(self):
        bad = np.array(((0, 1, 2), (0, 1, 3), (0, 1, 4)))
        with self.assertRaises(ValueError):
            extract_boundary_loops(bad)

    def test_disconnected_pieces_are_not_mistaken_for_three_hole_boundaries(self):
        self.rest = np.array([(x + offset, y, 0) for offset in (-2, 0, 2)
                              for x, y in ((0, 0), (.5, 0), (.5, .5), (0, .5))])
        self.faces = np.array([(a + offset, b + offset, c + offset) for offset in (0, 4, 8)
                               for a, b, c in ((0, 1, 2), (0, 2, 3))])
        result = self.analyze()
        self.assertFalse(result["threading_passed"])
        self.assertFalse(result["topology"]["passed"])
        self.assertEqual(result["topology"]["connected_component_count"], 3)


if __name__ == "__main__":
    unittest.main()
