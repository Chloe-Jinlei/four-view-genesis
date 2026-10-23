"""Analytical single-contact regression tests; no Genesis scene is created."""
import unittest
import runtime
import numpy as np
from contact_friction import coulomb_tangent_correction


class ContactFrictionTests(unittest.TestCase):
    def test_zero_friction_is_exact_identity(self):
        for d, n in [([1, 2, 3], [0, 0, .1]), ([0, 0, 0], [0, 0, 0])]:
            np.testing.assert_array_equal(coulomb_tangent_correction(d, n, 0, 0), np.zeros(3))

    def test_no_compression_means_no_friction(self):
        np.testing.assert_array_equal(coulomb_tangent_correction([1, 2, 3], [0, 0, 0], 1, 1), np.zeros(3))

    def test_static_friction_stops_shallow_incline(self):
        angle = np.deg2rad(15)
        normal = np.array([0, -np.sin(angle), np.cos(angle)])
        gravity_displacement = np.array([0, 0, -9.81]) * .01 ** 2
        dn = -np.dot(gravity_displacement, normal) * normal
        projected = gravity_displacement + dn
        correction = coulomb_tangent_correction(projected, dn, .30, .20)
        np.testing.assert_allclose(projected + correction, 0, atol=1e-18)
        self.assertLess(np.linalg.norm(correction), .30 * np.linalg.norm(dn))

    def test_kinetic_incline_matches_coulomb_acceleration(self):
        angle = np.deg2rad(35)
        normal = np.array([0, -np.sin(angle), np.cos(angle)])
        gravity_displacement = np.array([0, 0, -9.81]) * .01 ** 2
        dn = -np.dot(gravity_displacement, normal) * normal
        projected = gravity_displacement + dn
        correction = coulomb_tangent_correction(projected, dn, .30, .20)
        expected = 9.81 * (np.sin(angle) - .20 * np.cos(angle)) * .01 ** 2
        self.assertAlmostEqual(np.linalg.norm(projected + correction), expected, places=14)
        self.assertAlmostEqual(np.linalg.norm(correction), .20 * np.linalg.norm(dn), places=14)
        self.assertAlmostEqual(np.dot(correction, normal), 0, places=14)

    def test_friction_cannot_reverse_sliding_or_affect_normal_motion(self):
        displacement = np.array([.2, .1, .3])
        correction = coulomb_tangent_correction(displacement, [0, 0, .1], .4, .3)
        self.assertEqual(correction[2], 0)
        self.assertLessEqual(np.linalg.norm(correction), .03 + 1e-15)
        self.assertGreater(np.dot(displacement[:2], (displacement + correction)[:2]), 0)

    def test_invalid_coefficients_rejected(self):
        for a, b in [(-1, 0), (.1, .2), (np.nan, 0), (1, np.inf)]:
            with self.subTest(mu=(a, b)), self.assertRaises(ValueError):
                coulomb_tangent_correction([0, 0, 0], [0, 0, 1], a, b)


if __name__ == '__main__':
    unittest.main()
