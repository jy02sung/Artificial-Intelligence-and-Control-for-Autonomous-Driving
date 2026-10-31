"""Controlled dangerous fixtures, never part of collected demonstration data."""
import tempfile
import unittest
from pathlib import Path
import traci
import xml.etree.ElementTree as ET
import sumolib
from build import binary, HERE
from collect import snapshot
from rule_policy import RulePolicy

class SumoRuleTests(unittest.TestCase):
    def test_split_routes_and_geometry_preserved(self):
        net=sumolib.net.readNet(str(HERE/'rules_v2/network.net.xml'))
        baseline=sumolib.net.readNet(str(HERE/'toegyewon.net.xml'))
        for route in ET.parse(HERE/'rules_v2/routes.rou.xml').getroot().findall('route'):
            ids=route.get('edges').split()
            for a,b in zip(ids,ids[1:]):self.assertIn(net.getEdge(b),net.getEdge(a).getOutgoing())
        for i in range(4):
            a=net.getLane('170059994.dashed_'+str(i));b=net.getLane('170059994.solid_'+str(i))
            self.assertAlmostEqual(a.getLength()+b.getLength(),baseline.getLane('170059994_'+str(i)).getLength())
            self.assertEqual(a.getShape()[-1],b.getShape()[0])

    def exercise(self,blocked=False,risky=True):
        tmp=tempfile.TemporaryDirectory()
        traci.start([binary('sumo'),'-n',str(HERE/'rules_v2/network.net.xml'),
                     '--step-length','.2','--no-step-log','true','--duration-log.disable','true'])
        policy=RulePolicy(123,Path(tmp.name)/'events.jsonl')
        try:
            traci.route.add('test',['86198713','86198719','86198728','86198725','86198709','37927509','206426298','844484877'])
            traci.vehicle.add('ego','test',departLane='1',departPos='40',departSpeed='0')
            traci.vehicle.add('leader','test',departLane='1',departPos='80',departSpeed='0')
            if blocked:
                traci.vehicle.add('left','test',departLane='2',departPos='50',departSpeed='0')
                traci.vehicle.add('right','test',departLane='0',departPos='50',departSpeed='0')
            traci.simulationStep();snapshot() # establish subscriptions
            traci.vehicle.moveTo('ego','86198713_1',50)
            traci.vehicle.moveTo('leader','86198713_1',63 if risky else 110)
            # Deliberately inject a close stationary leader to test the supervisor.
            traci.vehicle.setPreviousSpeed('ego',20)
            traci.vehicle.setSpeed('leader',0)
            states={v:dict(edge=traci.vehicle.getRoadID(v),lane=traci.vehicle.getLaneIndex(v),
                          speed=traci.vehicle.getSpeed(v),position=traci.vehicle.getLanePosition(v))
                    for v in traci.vehicle.getIDList()}
            decision=policy.before_step(states)
            traci.simulationStep()
            lane=traci.vehicle.getLaneIndex('ego')
            return decision,lane
        finally:
            policy.close();traci.close();tmp.cleanup()

    def test_emergency_crosses_solid_with_empty_target(self):
        d,lane=self.exercise()
        self.assertTrue(d['emergency'],d)
        self.assertNotEqual(lane,1)

    def test_blocked_adjacent_lanes_only_brake(self):
        d,lane=self.exercise(blocked=True)
        self.assertTrue(d['brake'],d)
        self.assertFalse(d['emergency'],d)
        self.assertEqual(lane,1)

    def test_safe_gap_does_not_enable_exception(self):
        d,lane=self.exercise(risky=False)
        self.assertFalse(d['emergency'],d)
        self.assertEqual(lane,1)

if __name__=='__main__':unittest.main()
