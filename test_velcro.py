"""Deterministic tests for strip-contact selection; no Genesis scene needed.

Run with the installed Genesis Python: ``python -m unittest test_velcro``.
"""
import unittest

import runtime  # Configure local caches/headless imports before Genesis.
import numpy as np

from velcro import boundary_strips, select_nearest_contacts


class NearestContactSelectionTests(unittest.TestCase):
    def select(self, positions, velocities=None, left=(0, 1), right=(2, 3), **kw):
        positions = np.asarray(positions, dtype=float)
        velocities = np.zeros_like(positions) if velocities is None else velocities
        return select_nearest_contacts(
            positions, velocities, left, right, max_pairs=kw.pop('max_pairs', 8), **kw,
        )

    def test_crossed_material_order_matches_physical_contacts(self):
        positions = [[0, 0, 0], [0, .08, 0], [.011, .08, 0], [.012, 0, 0]]
        np.testing.assert_array_equal(self.select(positions), [[1, 2], [0, 3]])

    def test_distance_and_relative_speed_are_both_required(self):
        positions = [[0, 0, 0], [0, .08, 0], [.021, 0, 0], [.012, .08, 0]]
        velocities = np.zeros((4, 3))
        velocities[3, 0] = .101
        self.assertEqual(self.select(positions, velocities).shape, (0, 2))
        velocities[:] = [2.0, 0, 0]  # Common translation is not relative motion.
        np.testing.assert_array_equal(self.select(positions, velocities), [[1, 3]])

    def test_distance_limit_cannot_be_increased_above_twenty_mm(self):
        with self.assertRaises(ValueError):
            self.select(np.zeros((4, 3)), max_distance=.020001)

    def test_reserved_and_disjoint_endpoints(self):
        positions = [[0, 0, 0], [.001, 0, 0], [.010, 0, 0], [.012, 0, 0]]
        np.testing.assert_array_equal(self.select(positions, reserved=[1]), [[0, 2]])
        result = self.select(positions)
        self.assertEqual(len(np.unique(result)), result.size)
        np.testing.assert_array_equal(result, [[1, 2], [0, 3]])

    def test_deterministic_ties_and_capacity(self):
        positions = np.zeros((4, 3))
        np.testing.assert_array_equal(self.select(positions, left=(1, 0), right=(3, 2)), [[0, 2], [1, 3]])
        np.testing.assert_array_equal(self.select(positions, max_pairs=1), [[0, 2]])
        self.assertEqual(self.select(positions, max_pairs=0).shape, (0, 2))
        self.assertEqual(self.select(positions, left=()).shape, (0, 2))

    def test_inputs_are_not_mutated(self):
        positions = np.array([[0, 0, 0], [0, .08, 0], [.011, .08, 0], [.012, 0, 0]])
        velocities = np.zeros_like(positions)
        before = positions.copy(), velocities.copy()
        self.select(positions, velocities)
        np.testing.assert_array_equal(positions, before[0])
        np.testing.assert_array_equal(velocities, before[1])

    def test_invalid_groups_and_data_rejected(self):
        for kw in ({'right': (1, 3)}, {'left': (-1, 0)}, {'left': (0.0, 1.0)}, {'max_pairs': -1}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                self.select(np.zeros((4, 3)), **kw)
        with self.assertRaises(ValueError):
            self.select(np.full((4, 3), np.nan))

    def test_full_width_strips_are_material_ids(self):
        x = [-.5, -.48, -.47, 0, .47, .48, .5]
        initial = np.column_stack((x, np.zeros((7, 2))))
        wide = boundary_strips(initial)
        narrow = boundary_strips(initial, width=.012)
        np.testing.assert_array_equal(wide[0], [0, 1])
        np.testing.assert_array_equal(wide[1], [5, 6])
        self.assertGreater(len(wide[0]), len(narrow[0]))
        with self.assertRaises(ValueError):
            boundary_strips(initial, width=.5)


if __name__ == '__main__':
    unittest.main()
