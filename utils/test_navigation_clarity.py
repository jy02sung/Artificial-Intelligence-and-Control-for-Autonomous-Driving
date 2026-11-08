import unittest
from env.navigation_observation import RouteNavigation
from env.navigation_clarity import validity_features, second_constraint_features


class ValidityTests(unittest.TestCase):
    def test_second_branch_is_visible_before_first(self):
        nav = RouteNavigation({'a':3,'b':3,'c':2}, [('a','b',0,0),('a','b',1,1),('b','c',1,0)], {'a':50,'b':60,'c':50})
        result = second_constraint_features(nav,['a','b','c'],0,40)
        self.assertAlmostEqual(result[0],100/300)
        self.assertEqual(result[1:7],[-1.,1.,-1.,0.,0.,0.])
        self.assertEqual(result[7],1.)

    def test_no_future_constraint_and_internal_are_explicitly_absent(self):
        nav = RouteNavigation({'a':2,'b':2}, [('a','b',0,0)], {'a':50,'b':50})
        self.assertEqual(second_constraint_features(nav,['a','b'],0,40),[-1.,0.,0.,0.,0.,0.,0.,-1.])
        self.assertEqual(second_constraint_features(nav,['a','b'],0,40,True),[-1.,0.,0.,0.,0.,0.,0.,-1.])

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
