import copy
import unittest
import xml.etree.ElementTree as ET
import numpy as np
from comparison_setup import configure_ego, freeze_demand, demand_hash, split_for
from compare_controllers import improvement, episode_metrics

class ComparisonTests(unittest.TestCase):
    def root(self):
        return ET.fromstring('<routes><vType id="ego_baseline" carFollowModel="IDM"/><vType id="human"/><vType id="car_idm"/><vTypeDistribution id="background" vTypes="human car_idm" probabilities="0.5 0.5"/><route id="r" edges="a b"/><flow id="f" type="background" route="r" begin="0" end="30" vehsPerHour="360" departLane="best" departSpeed="max"/><vehicle id="ego" type="ego_baseline" route="r" depart="10"/></routes>')
    def test_only_model_changes_and_same_demand(self):
        roots=[]
        for model in ['Krauss','IDM','EIDM','ACC']:
            root=self.root();configure_ego(root,model);freeze_demand(root,100)
            roots.append(root)
        self.assertEqual(len({demand_hash(r) for r in roots}),1)
        attrs=[]
        for r in roots:
            a=dict(r.find("vType[@id='ego_baseline']").attrib);a.pop('carFollowModel');attrs.append(a)
        self.assertTrue(all(a==attrs[0] for a in attrs))
        self.assertEqual([v.get('depart') for v in roots[0].findall('vehicle')],['0','10','10','20'])
        self.assertFalse(roots[0].findall('flow'))
    def test_split_is_episode_based(self):
        self.assertEqual([split_for(i) for i in range(5)],['train','train','train','validation','test'])
    def test_zero_reference_has_no_percent(self):
        self.assertIsNone(improvement(0,0))
        self.assertEqual(improvement(100,80),20)
        self.assertEqual(improvement(100,120),-20)
    def test_failure_penalty_and_no_cross_episode_jerk(self):
        d={'obs':np.zeros((4,54)), 'action_applied':np.array([[0,0],[-4,0],[-4,1],[0,0]]),
           'action_valid':np.array([[1,1],[1,1],[1,1],[0,0]],bool)}
        e={'episode':0,'end_reason':'time_limit','steps':4,'ego_collisions':0,'ego_teleports':0}
        m=episode_metrics(e,d,np.arange(4),.2,300)
        self.assertEqual(m['penalized_time_s'],300)
        self.assertEqual(m['hard_brake_events'],1)
        self.assertAlmostEqual(m['jerk_rms'],(400/2)**.5)
        self.assertAlmostEqual(m['stopped_s'],.8)

if __name__=='__main__':unittest.main()
