import unittest

import pandas as pd

from backtesting import (
    count_same_day_time_gaps,
    evaluate_backtest,
    filter_combined_setups,
    summarize_trade_journal,
)


def make_setup(direction="BUY", signal_index=0, target_multiple=2):
    entry = 100.0
    stop = 99.0 if direction == "BUY" else 101.0
    target = (
        entry + target_multiple
        if direction == "BUY"
        else entry - target_multiple
    )
    return {
        "direction": direction,
        "entry": entry,
        "stop_loss": stop,
        "profit_target": target,
        "signal_index": signal_index,
        "signal_time": pd.Timestamp("2026-01-01T00:00:00Z"),
    }


class BacktestTests(unittest.TestCase):
    def test_counts_wins_losses_and_expectancy_in_r(self):
        candles = pd.DataFrame([
            {"High": 100, "Low": 100},
            {"High": 102, "Low": 100},
            {"High": 100, "Low": 98},
        ])
        summary = evaluate_backtest(
            candles,
            [make_setup(signal_index=0), make_setup(signal_index=1)],
        )
        self.assertEqual(summary["completed"], 2)
        self.assertEqual(summary["wins"], 1)
        self.assertEqual(summary["losses"], 1)
        self.assertAlmostEqual(summary["win_rate"], 0.5)
        self.assertAlmostEqual(summary["expectancy_r"], 0.5)
        self.assertAlmostEqual(summary["profit_factor"], 2.0)

    def test_candle_touching_both_levels_is_conservatively_a_loss(self):
        candles = pd.DataFrame([
            {"High": 100, "Low": 100},
            {"High": 102, "Low": 98},
        ])
        summary = evaluate_backtest(candles, [make_setup()])
        self.assertEqual(summary["ambiguous"], 1)
        self.assertEqual(summary["losses"], 1)
        self.assertEqual(summary["trades"][0]["Gross (R)"], -1)
        self.assertEqual(summary["trades"][0]["Net (R)"], -1)

    def test_trade_journal_records_stop_exit_details(self):
        exit_time = pd.Timestamp("2026-01-01T00:01:00Z")
        candles = pd.DataFrame([
            {"High": 100, "Low": 100, "Time": pd.Timestamp("2026-01-01T00:00:00Z")},
            {"High": 100.5, "Low": 98, "Time": exit_time},
        ])

        trade = evaluate_backtest(candles, [make_setup()])["trades"][0]

        self.assertEqual(trade["Exit Type"], "STOP_LOSS")
        self.assertEqual(trade["Exit Price"], 99)
        self.assertEqual(trade["Exit Time"], exit_time)
        self.assertEqual(trade["Bars to Exit"], 1)

    def test_unresolved_and_unclosed_signals_are_not_in_win_rate(self):
        candles = pd.DataFrame([
            {"High": 100, "Low": 100},
            {"High": 100.5, "Low": 99.5},
        ])
        summary = evaluate_backtest(candles, [make_setup()])
        self.assertEqual(summary["open"], 1)
        self.assertEqual(summary["completed"], 0)
        self.assertIsNone(summary["win_rate"])
        self.assertTrue(summary["sample_warning"])

    def test_skips_new_signals_while_a_prior_trade_is_open(self):
        candles = pd.DataFrame([
            {"High": 100, "Low": 100},
            {"High": 100.5, "Low": 99.5},
        ])
        summary = evaluate_backtest(
            candles,
            [make_setup(signal_index=0), make_setup(signal_index=1)],
        )
        self.assertEqual(summary["open"], 1)
        self.assertEqual(summary["skipped_overlapping"], 1)

    def test_costs_are_deducted_from_net_results(self):
        candles = pd.DataFrame([
            {"High": 100, "Low": 100},
            {"High": 102, "Low": 100},
        ])
        summary = evaluate_backtest(
            candles,
            [make_setup()],
            round_trip_cost_r=0.25,
        )
        self.assertEqual(summary["wins"], 1)
        self.assertAlmostEqual(summary["expectancy_r"], 1.75)
        self.assertAlmostEqual(summary["net_win_rate"], 1.0)

    def test_split_window_does_not_use_candles_after_test_end(self):
        candles = pd.DataFrame([
            {"High": 100, "Low": 100},
            {"High": 100.5, "Low": 99.5},
            {"High": 102, "Low": 100},
        ])
        summary = evaluate_backtest(
            candles,
            [make_setup()],
            start_index=0,
            end_index=2,
        )
        self.assertEqual(summary["completed"], 0)
        self.assertEqual(summary["open"], 1)

    def test_counts_intraday_timestamp_gap_pairs(self):
        candles = pd.DataFrame({
            "Time": pd.to_datetime([
                "2026-01-05T10:00:00Z",
                "2026-01-05T10:01:00Z",
                "2026-01-05T10:04:00Z",
                "2026-01-06T10:00:00Z",
            ], utc=True)
        })
        self.assertEqual(count_same_day_time_gaps(candles, 60), 1)

    def test_invalid_setup_is_skipped(self):
        candles = pd.DataFrame([{"High": 100, "Low": 100}])
        setup = make_setup()
        setup["stop_loss"] = setup["entry"]
        wrong_target = make_setup()
        wrong_target["profit_target"] = 99
        summary = evaluate_backtest(candles, [setup, wrong_target])
        self.assertEqual(summary["trades"], [])

    def test_combined_filter_requires_profile_confirmation(self):
        candles = pd.DataFrame([
            {"Close": 100},
            {"Close": 103},
            {"Close": 97},
        ])
        setups = [
            make_setup("BUY", signal_index=0),
            make_setup("BUY", signal_index=1),
            make_setup("SELL", signal_index=2),
        ]

        def load_profile(_signal_time):
            return {"profile": {"vah": 102, "val": 98}}

        combined, unavailable = filter_combined_setups(
            candles,
            setups,
            load_profile,
        )
        self.assertEqual([setup["direction"] for setup in combined], ["BUY", "SELL"])
        self.assertEqual(unavailable, 0)

    def test_combined_filter_excludes_setup_without_historical_profile(self):
        candles = pd.DataFrame([{"Close": 100}])
        combined, unavailable = filter_combined_setups(
            candles,
            [make_setup()],
            lambda _signal_time: {"profile": None},
        )
        self.assertEqual(combined, [])
        self.assertEqual(unavailable, 1)

    def test_trade_journal_flags_repeated_stop_losses_as_a_learning_pattern(self):
        trades = [
            {"Direction": "BUY", "Exit Type": "STOP_LOSS", "Result": "LOSS"},
            {"Direction": "BUY", "Exit Type": "STOP_LOSS", "Result": "LOSS"},
            {"Direction": "SELL", "Exit Type": "TARGET", "Result": "WIN"},
            {"Direction": "BUY", "Exit Type": "AMBIGUOUS", "Result": "AMBIGUOUS — counted as loss"},
        ]
        summary = summarize_trade_journal(trades)
        self.assertEqual(summary["stops_hit"], 2)
        self.assertEqual(summary["targets_hit"], 1)
        self.assertEqual(summary["ambiguous"], 1)
        self.assertIn("BUY", summary["most_common_mistake"])


if __name__ == "__main__":
    unittest.main()
