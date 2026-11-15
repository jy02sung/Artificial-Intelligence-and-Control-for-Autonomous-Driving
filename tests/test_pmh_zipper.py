import unittest

from utils import pmh_zipper_network as zn
from utils.pmh_zipper_audit import find_merges
from utils.pmh_zipper_scenario import measured_counts


class ZipperNetworkTests(unittest.TestCase):
    def test_only_lane_drop_nodes_change(self):
        report = zn.validate(zn.SOURCE, zn.OUTPUT)
        self.assertEqual(report['weave_170059994_length'], '51.07')
        self.assertEqual(len(report['added_connections']), 3)
        self.assertEqual(len(report['zipper_junctions']), 6)

    def test_every_merge_is_a_zipper_junction(self):
        merges = find_merges(zn.OUTPUT)
        self.assertEqual(len(merges), 6)
        self.assertTrue(all(m['kind'] == 'junction' and m['node_type'] == 'zipper' for m in merges))

    def test_source_network_still_has_lane_drops(self):
        kinds = sorted(m['kind'] for m in find_merges(zn.SOURCE))
        self.assertEqual(kinds, ['junction'] * 3 + ['lane_drop'] * 3)


class MeasuredDemandTests(unittest.TestCase):
    def test_low_exit25_matches_documented_total(self):
        counts, provenance = measured_counts('low_exit25')
        self.assertEqual(sum(counts.values()), 13949)
        self.assertEqual(provenance['route_06'], 'observed_mainline_envelope_x_assumed_turn_share')


if __name__ == '__main__':
    unittest.main()
