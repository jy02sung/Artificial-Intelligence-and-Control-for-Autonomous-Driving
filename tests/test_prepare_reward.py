import unittest
from env.prepare_reward import PrepareCredit


class PrepareTests(unittest.TestCase):
    def test_progress_only_and_no_oscillation_farming(self):
        r = PrepareCredit()
        self.assertEqual(r.award('a', 'b', [0], 3, 2, 250), .1)
        self.assertEqual(r.award('a', 'b', [0], 2, 3, 240), 0)
        self.assertEqual(r.award('a', 'b', [0], 3, 2, 230), 0)
        self.assertEqual(r.award('a', 'b', [0], 2, 1, 220), .1)

    def test_range_and_compatible_lanes(self):
        r = PrepareCredit()
        self.assertEqual(r.award('a', 'b', [0], 2, 1, 301), 0)
        self.assertEqual(r.award('a', 'b', [], 2, 1, 100), 0)
        self.assertEqual(r.award('a', 'b', [0,1], 0, 1, 100), 0)
        self.assertEqual(r.award('a', 'b', [0], 2, 1, 300), .1)


if __name__ == '__main__':
    unittest.main()
