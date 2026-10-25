import unittest

from volume_profile import build_volume_profile


class VolumeProfileTests(unittest.TestCase):
    def test_point_of_control_and_seventy_percent_value_area(self):
        profile = build_volume_profile(
            [
                {"price": 1.00, "tick_count": 2},
                {"price": 1.01, "tick_count": 10},
                {"price": 1.02, "tick_count": 3},
                {"price": 1.03, "tick_count": 1},
            ],
            bin_size=0.01,
        )
        self.assertAlmostEqual(profile["poc"], 1.015)
        self.assertAlmostEqual(profile["val"], 1.01)
        self.assertAlmostEqual(profile["vah"], 1.03)
        self.assertEqual(profile["total_ticks"], 16)

    def test_identifies_local_high_and_low_nodes(self):
        profile = build_volume_profile(
            [
                {"price": 10, "tick_count": 1},
                {"price": 11, "tick_count": 8},
                {"price": 12, "tick_count": 1},
                {"price": 13, "tick_count": 1},
                {"price": 14, "tick_count": 6},
                {"price": 15, "tick_count": 1},
            ],
            bin_size=1,
            value_area_fraction=0.7,
        )
        node_types = {node["type"] for node in profile["nodes"]}
        self.assertEqual(node_types, {"HVN", "LVN"})
        self.assertIn(
            {"type": "LVN", "price": 12.5, "tick_count": 1},
            profile["nodes"],
        )

    def test_empty_invalid_and_nonpositive_rows_are_ignored(self):
        self.assertIsNone(build_volume_profile([], bin_size=1))
        profile = build_volume_profile(
            [
                {"price": float("nan"), "tick_count": 2},
                {"price": 10, "tick_count": 0},
                {"price": 11, "tick_count": 3},
            ],
            bin_size=1,
        )
        self.assertEqual(profile["total_ticks"], 3)
        self.assertEqual(profile["ignored_rows"], 2)

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            build_volume_profile([], bin_size=0)
        with self.assertRaises(ValueError):
            build_volume_profile([], bin_size=1, value_area_fraction=0)

    def test_equal_poc_counts_resolve_to_lower_price_bin(self):
        profile = build_volume_profile(
            [{"price": 1, "tick_count": 5}, {"price": 2, "tick_count": 5}],
            bin_size=1,
        )
        self.assertEqual(profile["poc"], 1.5)


if __name__ == "__main__":
    unittest.main()
