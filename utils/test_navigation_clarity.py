import unittest
from env.navigation_observation import RouteNavigation
from env.navigation_clarity import validity_features


class ValidityTests(unittest.TestCase):
    def test_valid_zero_and_unknown_zero_differ(self):
        valid = dict(current_targets=[1], prepare_targets=[1])
        unknown = dict(current_targets=[], prepare_targets=[])
        self.assertEqual(RouteNavigation.offset(1, [1]), RouteNavigation.offset(1, []))
        self.assertEqual(validity_features(valid), [1., 1.])
        self.assertEqual(validity_features(unknown), [-1., -1.])

    def test_current_known_preparation_unknown(self):
        self.assertEqual(validity_features(dict(current_targets=[0, 1], prepare_targets=[])), [1., -1.])


if __name__ == '__main__':
    unittest.main()
