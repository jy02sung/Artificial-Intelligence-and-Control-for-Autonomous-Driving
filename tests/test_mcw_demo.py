import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from utils.mcw_demo import CROSS_DEPART_LANES, DEMO_OVERTAKE_EXTENSIONS, _shadow_quad_from_box
from utils.mcw_train import ROUTES, prepare_scenario


class McwDemoTests(unittest.TestCase):
    def test_cross_depart_lanes_covers_all_routes(self):
        self.assertEqual(set(CROSS_DEPART_LANES.keys()), set(ROUTES))
        self.assertEqual(CROSS_DEPART_LANES["route_13"], "4")
        self.assertEqual(CROSS_DEPART_LANES["route_05"], "3")

    def test_shadow_quad_projects_to_visibility_radius(self):
        quad = _shadow_quad_from_box(0.0, 0.0, 120.0, 0.0, 20.0, 0.0, 1.0, 1.0, 0.0, 4.5, 1.25)
        self.assertEqual(len(quad), 4)
        self.assertAlmostEqual((quad[1][0] ** 2 + quad[1][1] ** 2) ** 0.5, 120.0, places=4)
        self.assertAlmostEqual((quad[2][0] ** 2 + quad[2][1] ** 2) ** 0.5, 120.0, places=4)

    def test_demo_dense_has_real_shapes_gui_settings_and_slow_trucks(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = prepare_scenario("demo_dense", Path(tmp), color_by_type=True, gui_delay=55.0)
            gui_xml = ET.parse(str(cfg.parent / "highway.gui.xml")).getroot()
            self.assertEqual(gui_xml.find("scheme").get("name"), "real world")
            self.assertEqual(gui_xml.find("delay").get("value"), "55.0")
            cfg_root = ET.parse(str(cfg)).getroot()
            self.assertEqual(cfg_root.find("gui_only/gui-settings-file").get("value"), "highway.gui.xml")
            rou = ET.parse(str(cfg.parent / "highway.rou.xml")).getroot()
            vtypes = {vt.get("id"): vt for vt in rou.findall("vType")}
            self.assertEqual(vtypes["car_krauss"].get("guiShape"), "passenger/sedan")
            self.assertEqual(vtypes["car_eidm"].get("guiShape"), "passenger/hatchback")
            self.assertEqual(vtypes["car_acc"].get("guiShape"), "passenger/wagon")
            self.assertIn("truck_slow", vtypes)
            self.assertIn("bus_slow", vtypes)

    def test_demo_overtake_extends_straight_approaches_above_850m(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = prepare_scenario("demo_overtake", Path(tmp), color_by_type=True, gui_delay=55.0)
            net = ET.parse(str(cfg.parent / "network.net.xml")).getroot()
            lengths = {
                e.get("id"): max(float(l.get("length")) for l in e.findall("lane"))
                for e in net.findall("edge")
                if not e.get("function")
            }
            for eid in DEMO_OVERTAKE_EXTENSIONS:
                self.assertGreater(lengths[eid], 850.0)
            self.assertGreater(lengths["86198728"], 1000.0)
            self.assertGreater(lengths["52873356"], 1000.0)


if __name__ == "__main__":
    unittest.main()
