import unittest
import numpy as np
from env.route_learning_trials import reward_parts, compact_observation
class Components(unittest.TestCase):
    def test_split_and_scale(self):
        parts=reward_parts(.12,.02,5)
        self.assertAlmostEqual(parts['selection'],.5)
        self.assertAlmostEqual(parts['maintenance'],.1)
        self.assertAlmostEqual(sum(parts.values()),.6)
        self.assertAlmostEqual(reward_parts(-.1,0,1)['departure'],-.1)
    def test_compact_keeps_validity_and_direction_not_mask_cost(self):
        full=np.arange(63,dtype=np.float32)
        out=compact_observation(full)
        np.testing.assert_array_equal(out[:51],full[:51])
        np.testing.assert_array_equal(out[57:59],full[57:59])
        self.assertTrue((out[51:57]==0).all())
        self.assertTrue((out[59:]==0).all())
if __name__=='__main__':unittest.main()
