import unittest
from utils.conservative_update import with_learning_rate

class ConservativeUpdateTest(unittest.TestCase):
    def test_only_lr_changes_and_source_is_preserved(self):
        source={'hp':{'lr':.0003,'gamma':.99,'clip_eps':.2},'reward':{'collision':5}}
        output=with_learning_rate(source,.00003)
        self.assertEqual(source['hp']['lr'],.0003)
        self.assertEqual(output['hp'],{'lr':.00003,'gamma':.99,'clip_eps':.2})
        self.assertEqual(output['reward'],source['reward'])

    def test_invalid_rate_is_rejected(self):
        with self.assertRaises(ValueError):with_learning_rate({'hp':{'lr':.0003}},0)

if __name__=='__main__':unittest.main()
