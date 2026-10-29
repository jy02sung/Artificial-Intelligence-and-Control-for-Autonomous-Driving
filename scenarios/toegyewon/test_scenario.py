"""Network acceptance checks; run after build.py, then run.py for dynamic checks."""
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
import sumolib

HERE = Path(__file__).resolve().parent

class ScenarioTests(unittest.TestCase):
    def test_network_and_routes(self):
        net = sumolib.net.readNet(str(HERE / 'toegyewon.net.xml'))
        root = ET.parse(HERE / 'toegyewon.rou.xml').getroot()
        ego = root.findall("vehicle[@id='ego']")
        self.assertEqual(len(ego), 1)
        routes = {r.get('id'):r.get('edges').split() for r in root.findall('route')}
        for name, ids in routes.items():
            for a, b in zip(ids, ids[1:]):
                self.assertIn(net.getEdge(b), net.getEdge(a).getOutgoing(), (name,a,b))
        common = set(routes['nw_to_east']) & set(routes['south_to_west'])
        self.assertTrue(any(e.startswith('170059994') for e in common), common)
        self.assertTrue(any(e.startswith('170059988') for e in routes['nw_to_east']))
        self.assertTrue(any(e.startswith('170059992') for e in routes['south_to_west']))

    def test_weaving_requires_cross_lane_movement(self):
        root = ET.parse(HERE / 'toegyewon.net.xml').getroot()
        links = root.findall('connection')
        entering = [c for c in links if c.get('from')=='170059995' and c.get('to')=='170059994']
        east = [c for c in links if c.get('from')=='170059994' and c.get('to')=='1063527748']
        self.assertEqual({c.get('toLane') for c in entering},{'3'})
        self.assertNotIn('3',{c.get('fromLane') for c in east})
        upper_loop = [c for c in links if c.get('from')=='170059994' and c.get('to')=='844484876']
        self.assertEqual({c.get('fromLane') for c in upper_loop},{'3'})

    def test_bridge_crossing_is_not_an_at_grade_junction(self):
        net = sumolib.net.readNet(str(HERE / 'toegyewon.net.xml'))
        bridge = net.getEdge('170059996')
        for road in ['37927509','844484878#0','844484878#1']:
            main = net.getEdge(road)
            self.assertFalse({bridge.getFromNode(),bridge.getToNode()} & {main.getFromNode(),main.getToNode()})

if __name__ == '__main__':
    unittest.main()
