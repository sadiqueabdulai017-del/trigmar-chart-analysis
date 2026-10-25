"""Conservative closed-candle backtest summaries for confirmed setups."""

import math
from collections import Counter

import pandas as pd


def summarize_trade_journal(trades):
    """Extract the recurring mistake patterns from a completed trade journal."""
    if not trades:
        return {
            "stops_hit": 0,
            "targets_hit": 0,
            "ambiguous": 0,
            "most_common_mistake": "None",
            "direction_counts": {},
            "pattern": "No closed trades available yet.",
        }

    stops_hit = sum(
        1 for trade in trades
        if str(trade.get("Exit Type", "")).upper() == "STOP_LOSS"
    )
    targets_hit = sum(
        1 for trade in trades
        if str(trade.get("Exit Type", "")).upper() == "TARGET"
    )
    ambiguous = sum(
        1 for trade in trades
        if str(trade.get("Exit Type", "")).upper() == "AMBIGUOUS"
    )
    direction_counts = Counter()
    for trade in trades:
        if str(trade.get("Exit Type", "")).upper() == "STOP_LOSS":
            direction = str(trade.get("Direction", "UNKNOWN")).upper()
            direction_counts[direction] += 1

    most_common_mistake = (
        max(direction_counts.items(), key=lambda item: item[1])[0]
        if direction_counts
        else "None"
    )
    pattern = (
        "Repeated stop-loss failures are the dominant mistake."
        if stops_hit >= 2
        else "Stop management is still the main edge to improve."
    )

    return {
        "stops_hit": stops_hit,
        "targets_hit": targets_hit,
        "ambiguous": ambiguous,
        "most_common_mistake": most_common_mistake,
        "direction_counts": dict(direction_counts),
        "pattern": pattern,
    }


def evaluate_backtest(
    candles,
    setups,
    minimum_sample=30,
    start_index=0,
    end_index=None,
    round_trip_cost_r=0.0,
):
    """Evaluate setup outcomes using only candles that follow each signal bar.

    When a later candle touches both stop and target, the trade is counted as a
    loss because OHLC data cannot reveal which level was reached first.
    Only one trade may be open at a time; signals before the current trade exits
    are skipped.
    """
    if not math.isfinite(round_trip_cost_r) or round_trip_cost_r < 0:
        raise ValueError("round_trip_cost_r must be a finite, nonnegative value.")
    if end_index is None:
        end_index = len(candles)
    start_index = max(0, start_index)
    end_index = min(len(candles), end_index)
    if end_index < start_index:
        raise ValueError("end_index must not precede start_index.")

    trades = []
    ordered_setups = sorted(setups, key=lambda setup: setup["signal_index"])
    available_index = start_index
    skipped_overlapping = 0
    for setup in ordered_setups:
        signal_index = int(setup["signal_index"])
        direction = setup["direction"]
        entry = float(setup["entry"])
        stop = float(setup["stop_loss"])
        target = float(setup["profit_target"])
        risk = abs(entry - stop)
        if (
            direction not in ("BUY", "SELL")
            or not all(math.isfinite(value) for value in (entry, stop, target, risk))
            or risk <= 0
            or (
                direction == "BUY"
                and not stop < entry < target
            )
            or (
                direction == "SELL"
                and not target < entry < stop
            )
            or signal_index < 0
            or signal_index >= end_index
        ):
            continue
        if signal_index < start_index:
            continue
        if signal_index < available_index:
            skipped_overlapping += 1
            continue

        result = "OPEN"
        exit_type = "OPEN"
        exit_price = None
        gross_r = None
        exit_index = end_index
        for index in range(signal_index + 1, end_index):
            candle = candles.iloc[index]
            stop_hit = (
                float(candle["Low"]) <= stop
                if direction == "BUY"
                else float(candle["High"]) >= stop
            )
            target_hit = (
                float(candle["High"]) >= target
                if direction == "BUY"
                else float(candle["Low"]) <= target
            )
            if stop_hit and target_hit:
                result = "AMBIGUOUS — counted as loss"
                exit_type = "AMBIGUOUS"
                gross_r = -1.0
                exit_index = index
                break
            if target_hit:
                result = "WIN"
                exit_type = "TARGET"
                exit_price = target
                gross_r = abs(target - entry) / risk
                exit_index = index
                break
            if stop_hit:
                result = "LOSS"
                exit_type = "STOP_LOSS"
                exit_price = stop
                gross_r = -1.0
                exit_index = index
                break

        net_r = (
            gross_r - round_trip_cost_r
            if gross_r is not None
            else None
        )
        trades.append({
            "Signal Time": setup["signal_time"],
            "Direction": direction,
            "Entry": entry,
            "Stop Loss": stop,
            "Target": target,
            "Result": result,
            "Exit Type": exit_type,
            "Exit Price": exit_price,
            "Exit Time": (
                candles.iloc[exit_index]["Time"]
                if exit_type != "OPEN" and "Time" in candles
                else None
            ),
            "Bars to Exit": (
                exit_index - signal_index
                if exit_type != "OPEN"
                else max(0, end_index - signal_index - 1)
            ),
            "Gross (R)": gross_r,
            "Net (R)": net_r,
        })
        available_index = exit_index

    completed = [trade for trade in trades if trade["Gross (R)"] is not None]
    wins = [trade for trade in completed if trade["Gross (R)"] > 0]
    net_results = [trade["Net (R)"] for trade in completed]
    net_wins = [result for result in net_results if result > 0]
    net_losses = [result for result in net_results if result < 0]
    net_profit_r = sum(net_wins)
    net_loss_r = -sum(net_losses)
    equity = 0.0
    peak = 0.0
    max_drawdown_r = 0.0
    for result in net_results:
        equity += result
        peak = max(peak, equity)
        max_drawdown_r = max(max_drawdown_r, peak - equity)

    completed_count = len(completed)
    return {
        "trades": trades,
        "completed": completed_count,
        "open": len(trades) - completed_count,
        "skipped_overlapping": skipped_overlapping,
        "wins": len(wins),
        "losses": len(completed) - len(wins),
        "ambiguous": sum(
            trade["Result"] == "AMBIGUOUS — counted as loss"
            for trade in completed
        ),
        "win_rate": len(wins) / completed_count if completed_count else None,
        "net_win_rate": (
            len(net_wins) / completed_count if completed_count else None
        ),
        "round_trip_cost_r": round_trip_cost_r,
        "expectancy_r": (
            sum(net_results) / completed_count if completed_count else None
        ),
        "profit_factor": (
            net_profit_r / net_loss_r if net_loss_r else None
        ),
        "max_drawdown_r": max_drawdown_r,
        "sample_warning": completed_count < minimum_sample,
    }


def filter_combined_setups(candles, setups, profile_loader):
    """Keep breakouts confirmed beyond the preceding session's value area."""
    combined = []
    unavailable_profiles = 0
    for setup in setups:
        signal_index = int(setup["signal_index"])
        if signal_index < 0 or signal_index >= len(candles):
            continue
        signal_time = setup["signal_time"]
        profile_data = profile_loader(signal_time)
        profile = profile_data.get("profile") if profile_data else None
        if profile is None:
            unavailable_profiles += 1
            continue
        close = float(candles.iloc[signal_index]["Close"])
        confirmed_beyond_value = (
            close > profile["vah"]
            if setup["direction"] == "BUY"
            else close < profile["val"]
        )
        if confirmed_beyond_value:
            combined.append(setup)
    return combined, unavailable_profiles


def count_same_day_time_gaps(candles, granularity_seconds):
    """Count adjacent same-UTC-day candle pairs with a missing interval."""
    if granularity_seconds <= 0 or "Time" not in candles:
        raise ValueError("A positive timeframe and Time column are required.")
    if len(candles) < 2:
        return 0

    times = pd.to_datetime(candles["Time"], utc=True)

    gaps = 0
    for previous, current in zip(times.iloc[:-1], times.iloc[1:]):
        if previous.date() == current.date():
            if (current - previous).total_seconds() > granularity_seconds:
                gaps += 1
    return gaps
