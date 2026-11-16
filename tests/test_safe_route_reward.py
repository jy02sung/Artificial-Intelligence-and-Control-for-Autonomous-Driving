import unittest
from env.safe_route_reward_trials import unsafe_route_withholding


class SafeRouteRewardTest(unittest.TestCase):
    def test_safe_or_no_bonus_has_no_withholding(self):
        self.assertEqual(unsafe_route_withholding({'selection': .1, 'maintenance': .02}, 2, .199), 0)
        self.assertEqual(unsafe_route_withholding({'departure': -.1, 'lane': -.1}, 2, 1), 0)

    def test_only_scaled_positive_route_bonus_is_withheld(self):
        parts = {'selection': .1, 'maintenance': .03, 'departure': -.1, 'lane': -.05}
        self.assertAlmostEqual(unsafe_route_withholding(parts, 2, .2), .26)
        self.assertEqual(parts['departure'], -.1)
        self.assertEqual(parts['lane'], -.05)


if __name__ == '__main__':
    unittest.main()
