import unittest
from env.junction_lookahead_trials import visible_conflict


class JunctionLookaheadTest(unittest.TestCase):
    def test_physically_visible_parallel_vehicle_is_retained(self):
        self.assertFalse(visible_conflict(140, 40, 50, 50))
        self.assertTrue(visible_conflict(140, 40, 150, 50))

    def test_larger_map_horizon_does_not_enlarge_sensor_range(self):
        self.assertFalse(visible_conflict(100, 51, 150, 50))
        self.assertFalse(visible_conflict(151, 30, 150, 50))
        self.assertTrue(visible_conflict(150, 50, 150, 50))


if __name__ == '__main__':
    unittest.main()
