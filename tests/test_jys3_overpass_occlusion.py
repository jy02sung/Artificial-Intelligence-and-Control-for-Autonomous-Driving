"""The same sight line can pass through a truck's footprint on a different deck."""
import unittest
from unittest.mock import patch

from env import jys3_dagger_sensors as sensors


class OverpassVisibilityTest(unittest.TestCase):
    def blocked(self, ego_edge, truck_edge, target_edge, y=1189.0):
        tc = sensors.traci.constants
        sensor = object.__new__(sensors.DaggerSensors)
        sensor._occ_sensor_mask = True
        sensor._occ_subscribed = True
        sensor.ego_id = "ego"
        sensor.visibility = 120
        sensor.occ_truck_rate = 1

        def state(edge, x, kind):
            return {tc.VAR_POSITION: (x, y), tc.VAR_ANGLE: 90, tc.VAR_LENGTH: 5,
                    tc.VAR_WIDTH: 2, tc.VAR_ROAD_ID: edge, tc.VAR_VEHICLECLASS: kind}

        with patch.object(sensors.traci, "vehicle") as cars:
            cars.getContextSubscriptionResults.return_value = {
                "truck": state(truck_edge, 1660, "truck"),
                "target": state(target_edge, 1665, "passenger"),
            }
            cars.getPosition.return_value = (1650, y)
            cars.getAngle.return_value = 90
            cars.getRoadID.return_value = ego_edge
            sensor._update_occlusion_state()
            return sensor._is_occluded("target")

    def test_overpass_in_both_directions(self):
        main, ramp = "206426298", "170059996"
        self.assertFalse(self.blocked(main, ramp, main))
        self.assertFalse(self.blocked(ramp, main, ramp))

    def test_ordinary_occlusion_is_preserved(self):
        main, ramp = "206426298", "170059996"
        self.assertTrue(self.blocked(main, main, main))
        self.assertTrue(self.blocked(main, "other_merge_road", main))
        self.assertTrue(self.blocked(main, ramp, main, y=1300))
        self.assertTrue(self.blocked(main, ramp, ramp))
