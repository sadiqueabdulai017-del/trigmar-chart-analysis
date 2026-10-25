import unittest

import pandas as pd

from supply_demand import calculate_supply_demand_zones


def make_demand_series(structure_high=100.2, follow_up=None):
    rows = []
    for index in range(10):
        rows.append({
            "Time": pd.Timestamp("2026-01-05T00:00:00Z")
            + pd.Timedelta(minutes=index),
            "Open": 100.0,
            "High": structure_high if index == 9 else 100.2,
            "Low": 99.8,
            "Close": 100.0,
        })
    rows.extend([
        {
            "Time": pd.Timestamp("2026-01-05T00:10:00Z"),
            "Open": 100.0, "High": 100.05, "Low": 99.85, "Close": 99.9,
        },
        {
            "Time": pd.Timestamp("2026-01-05T00:11:00Z"),
            "Open": 99.9, "High": 100.55, "Low": 99.85, "Close": 100.5,
        },
        {
            "Time": pd.Timestamp("2026-01-05T00:12:00Z"),
            "Open": 100.5, "High": 101.15, "Low": 100.45, "Close": 101.1,
        },
        {
            "Time": pd.Timestamp("2026-01-05T00:13:00Z"),
            "Open": 101.1, "High": 101.75, "Low": 101.05, "Close": 101.7,
        },
    ])
    if follow_up:
        for offset, candle in enumerate(follow_up, start=14):
            rows.append({
                "Time": pd.Timestamp("2026-01-05T00:00:00Z")
                + pd.Timedelta(minutes=offset),
                **candle,
            })
    return pd.DataFrame(rows)


class SupplyDemandZoneTests(unittest.TestCase):
    def test_requires_displacement_to_break_prior_structure(self):
        zones, _ = calculate_supply_demand_zones(
            make_demand_series(structure_high=101.8)
        )
        self.assertFalse(zones)

    def test_tracks_retest_then_later_invalidation(self):
        zones, latest = calculate_supply_demand_zones(
            make_demand_series(follow_up=[
                {
                    "Open": 100.0, "High": 100.4,
                    "Low": 100.0, "Close": 100.3,
                },
                {
                    "Open": 100.0, "High": 100.1,
                    "Low": 99.7, "Close": 99.8,
                },
            ])
        )
        zone = next(zone for zone in zones if zone["origin_index"] == 10)
        self.assertEqual(zone["status"], "Invalidated")
        self.assertEqual(zone["touches"], 1)
        self.assertIsNone(latest["Demand"])

    def test_reports_strength_and_tracks_one_touch_episode(self):
        zones, latest = calculate_supply_demand_zones(
            make_demand_series(follow_up=[
                {
                    "Open": 100.0, "High": 100.4,
                    "Low": 100.0, "Close": 100.3,
                },
                {
                    "Open": 100.3, "High": 100.6,
                    "Low": 100.2, "Close": 100.5,
                },
            ])
        )
        zone = next(zone for zone in zones if zone["origin_index"] == 10)
        self.assertEqual(zone["status"], "Retest confirmed")
        self.assertEqual(zone["touches"], 1)
        self.assertGreaterEqual(zone["strength_score"], 0)
        self.assertLessEqual(zone["strength_score"], 100)
        self.assertIn(zone["strength"], {"Developing", "Moderate", "Strong"})
        self.assertIsNotNone(latest["Demand"])

    def test_rejects_weak_impulse_breaks_that_are_not_perfect_zones(self):
        rows = []
        for index in range(10):
            rows.append({
                "Time": pd.Timestamp("2026-01-05T00:00:00Z")
                + pd.Timedelta(minutes=index),
                "Open": 100.0,
                "High": 100.05,
                "Low": 99.95,
                "Close": 100.02,
            })
        rows.extend([
            {
                "Time": pd.Timestamp("2026-01-05T00:10:00Z"),
                "Open": 100.0, "High": 100.1, "Low": 99.9, "Close": 100.02,
            },
            {
                "Time": pd.Timestamp("2026-01-05T00:11:00Z"),
                "Open": 100.02, "High": 100.08, "Low": 99.98, "Close": 100.04,
            },
            {
                "Time": pd.Timestamp("2026-01-05T00:12:00Z"),
                "Open": 100.04, "High": 100.15, "Low": 100.0, "Close": 100.11,
            },
        ])
        zones, _ = calculate_supply_demand_zones(pd.DataFrame(rows))
        self.assertFalse(zones)


if __name__ == "__main__":
    unittest.main()
