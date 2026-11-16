import unittest
from utils.complete_saved_trial import validate_suite


class SavedEvaluationTest(unittest.TestCase):
    def rows(self):
        return [dict(route=f'route_{i:02d}', seed=seed, traffic=mode,
                     condition='future', checkpoint='policy-51200.pt', route_overrides=0)
                for i in range(1, 17) for mode in ('none', 'reference')
                for seed in ((79101,) if mode == 'none' else (79101, 79102, 79103))]

    def test_complete_matched_suite(self):
        validate_suite(self.rows(), 'future', 'policy-51200.pt', (79101, 79102, 79103))

    def test_rejects_partial_duplicate_and_wrong_seed(self):
        for change in ('partial', 'duplicate', 'seed'):
            rows = self.rows()
            if change == 'partial': rows.pop()
            elif change == 'duplicate': rows[-1] = rows[0].copy()
            else: rows[-1]['seed'] = 99999
            with self.subTest(change=change), self.assertRaises(AssertionError):
                validate_suite(rows, 'future', 'policy-51200.pt', (79101, 79102, 79103))


if __name__ == '__main__':
    unittest.main()
