import unittest

from uav_path_planning import Bounds3D, Environment3D, ObstacleBox, Path3D, evaluate_path


class ExactGeometryTests(unittest.TestCase):
    def setUp(self):
        self.box = ObstacleBox((2.0, 2.0, 2.0), (2.01, 3.0, 3.0))
        self.env = Environment3D(Bounds3D((0, 0, 0), (5, 5, 5)), [self.box])

    def test_thin_obstacle_is_not_missed_by_large_step(self):
        self.assertFalse(self.env.segment_is_free((0, 2.5, 2.5), (5, 2.5, 2.5), step=100.0))

    def test_tangent_contact_counts_as_collision(self):
        tangent = ObstacleBox((2, 2, 2), (3, 3, 3))
        self.assertTrue(tangent.intersects_segment((0, 2, 2), (5, 2, 2)))
        self.assertFalse(tangent.intersects_segment((0, 1, 1), (5, 1, 1)))

    def test_parallel_and_zero_length_segments(self):
        self.assertTrue(self.box.intersects_segment((2, 2.5, 2.5), (2, 2.5, 2.5)))
        self.assertFalse(self.box.intersects_segment((1, 1, 1), (1, 1, 1)))
        self.assertFalse(self.box.intersects_segment((1, 1, 2.5), (1, 4, 2.5)))

    def test_exact_segment_clearance(self):
        # Segment is one unit away from the box in x.
        self.assertAlmostEqual(self.box.segment_distance((0, 2.5, 2.5), (1, 2.5, 2.5)), 1.0)
        self.assertEqual(self.box.segment_distance((0, 2.5, 2.5), (5, 2.5, 2.5)), 0.0)

    def test_margin_expands_collision_and_metrics_use_exact_clearance(self):
        segment = ((0, 2.5, 2.5), (1.5, 2.5, 2.5))
        self.assertTrue(self.env.segment_is_free(*segment, safety_margin=0.4))
        self.assertFalse(self.env.segment_is_free(*segment, safety_margin=0.6))
        metrics = evaluate_path(Path3D(list(segment)), self.env, safety_margin=0.6)
        self.assertFalse(metrics.feasible)
        self.assertEqual(metrics.min_obstacle_distance, 0.0)

    def test_invalid_margin_rejected(self):
        with self.assertRaises(ValueError):
            self.env.is_free((1, 1, 1), safety_margin=-1)
        with self.assertRaises(ValueError):
            self.box.intersects_segment((0, 0, 0), (1, 1, 1), margin=float("nan"))


if __name__ == "__main__":
    unittest.main()
