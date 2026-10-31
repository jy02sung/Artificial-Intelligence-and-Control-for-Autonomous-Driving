import unittest
import xml.etree.ElementTree as ET
from traffic import make_routes, CONFIG, HERE, OUT

class TrafficTests(unittest.TestCase):
    def test_no_ego_and_long_demand(self):
        root=make_routes(1.,42)
        self.assertFalse(root.findall("vehicle[@id='ego']"))
        self.assertFalse(root.findall("vType[@id='ego_baseline']"))
        self.assertGreater(len(root.findall('vehicle')),1400)
        self.assertLess(max(float(v.get('depart')) for v in root.findall('vehicle')),CONFIG['demand_end_s'])
        self.assertEqual({t.get('carFollowModel') for t in root.findall('vType')},{'Krauss','IDM','EIDM','ACC'})
    def test_matched_explicit_demand(self):
        self.assertEqual(ET.tostring(make_routes(1.,42)),ET.tostring(make_routes(1.,42)))
        self.assertNotEqual(ET.tostring(make_routes(1.,42)),ET.tostring(make_routes(1.,43)))
    def test_only_merge_blocks_changed(self):
        old=ET.parse(HERE/'rules_v2/network.net.xml').getroot()
        new=ET.parse(OUT/'network.net.xml').getroot()
        for e in old.findall('edge'):
            if e.get('function')!='internal':
                self.assertEqual(ET.tostring(e),ET.tostring(new.find(f"edge[@id='{e.get('id')}']")))
        for ident in CONFIG['merge_nodes']:
            self.assertEqual(new.find(f"junction[@id='{ident}']").get('type'),'zipper')
        for ramp in ['797789364','170059992']:
            links=[c for c in new.findall('connection') if c.get('from')==ramp]
            self.assertTrue(links)
            self.assertTrue(all(c.get('state')=='Z' for c in links))
    def test_preview_never_teleports_and_runs_without_ego(self):
        cfg=ET.parse(OUT/'traffic.sumocfg').getroot()
        self.assertEqual(cfg.find('processing/time-to-teleport').get('value'),'-1')
        self.assertEqual(float(cfg.find('time/end').get('value')),2400)

if __name__=='__main__':unittest.main()
