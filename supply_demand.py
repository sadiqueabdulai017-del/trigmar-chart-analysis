"""Supply and demand zone detection from closed OHLC candles."""

import pandas as pd


def calculate_supply_demand_zones(df, granularity_seconds=300):
    """Return high-quality zones only when the impulse is strong and clean."""
    if granularity_seconds <= 0:
        raise ValueError("granularity_seconds must be positive.")
    if df is None or df.empty:
        return [], {"Demand": None, "Supply": None}

    required_columns = {"Time", "Open", "High", "Low", "Close"}
    if not required_columns.issubset(df.columns):
        return [], {"Demand": None, "Supply": None}

    df = df.copy().sort_values("Time").reset_index(drop=True)
    zones = []
    average_range_period = 10
    impulse_length = 3

    for end_index in range(average_range_period + impulse_length, len(df)):
        origin_index = end_index - impulse_length
        origin = df.iloc[origin_index]
        impulse = df.iloc[origin_index + 1:end_index + 1]
        base = df.iloc[origin_index - average_range_period:origin_index]
        if len(base) < average_range_period:
            continue

        prior_ranges = base["High"] - base["Low"]
        average_range = float(prior_ranges.mean())
        if average_range <= 0:
            continue

        ranges = impulse["High"] - impulse["Low"]
        if (ranges <= 0).any():
            continue

        displacement = float(impulse.iloc[-1]["Close"] - origin["Open"])
        displacement_ratio = abs(displacement) / average_range
        body_efficiency = float(
            ((impulse["Close"] - impulse["Open"]).abs() / ranges).mean()
        )
        if body_efficiency < 0.72 or displacement_ratio < 1.8:
            continue

        bullish = (
            (impulse["Close"] > impulse["Open"])
            & ((impulse["Close"] - impulse["Open"]) / ranges >= 0.6)
        ).all()
        bearish = (
            (impulse["Close"] < impulse["Open"])
            & ((impulse["Open"] - impulse["Close"]) / ranges >= 0.6)
        ).all()
        if not bullish and not bearish:
            continue

        strength_score = round(
            60 * min(displacement_ratio / 3, 1)
            + 40 * min(body_efficiency / 0.9, 1)
        )
        if strength_score < 70:
            continue
        strength = (
            "Strong" if strength_score >= 85
            else "Moderate" if strength_score >= 70
            else "Developing"
        )

        broke_structure_up = float(impulse.iloc[-1]["Close"]) > float(
            base["High"].max()
        )
        broke_structure_down = float(impulse.iloc[-1]["Close"]) < float(
            base["Low"].min()
        )

        if (
            bullish
            and float(origin["Close"]) < float(origin["Open"])
            and displacement >= average_range * 1.5
            and broke_structure_up
        ):
            direction = "Demand"
        elif (
            bearish
            and float(origin["Close"]) > float(origin["Open"])
            and -displacement >= average_range * 1.5
            and broke_structure_down
        ):
            direction = "Supply"
        else:
            continue

        zone = {
            "direction": direction,
            "origin_index": origin_index,
            "start_time": origin["Time"],
            "low": float(origin["Low"]),
            "high": float(origin["High"]),
            "impulse_end": end_index,
            "strength_score": strength_score,
            "strength": strength,
            "status": "Active",
            "retest_index": None,
            "confirmed_at": None,
            "touches": 0,
        }
        touching = False
        for index in range(end_index + 1, len(df)):
            candle = df.iloc[index]
            invalidated = (
                float(candle["Close"]) < zone["low"]
                if direction == "Demand"
                else float(candle["Close"]) > zone["high"]
            )
            if invalidated:
                zone["status"] = "Invalidated"
                break

            touched = (
                float(candle["Low"]) <= zone["high"]
                and float(candle["High"]) >= zone["low"]
            )
            if touched and not touching:
                zone["touches"] += 1
            touching = touched
            if touched and zone["status"] == "Active":
                zone["status"] = "Tested"

            confirm_condition = (
                direction == "Demand"
                and touched
                and float(candle["Close"]) > zone["high"]
                and float(candle["Close"]) > float(candle["Open"])
            ) or (
                direction == "Supply"
                and touched
                and float(candle["Close"]) < zone["low"]
                and float(candle["Close"]) < float(candle["Open"])
            )
            if confirm_condition and zone["status"] != "Retest confirmed":
                zone["status"] = "Retest confirmed"
                zone["retest_index"] = index
                zone["confirmed_at"] = pd.Timestamp(candle["Time"]) + pd.Timedelta(
                    seconds=granularity_seconds
                )
        zones.append(zone)

    active_zones = [zone for zone in zones if zone["status"] != "Invalidated"]
    latest_by_direction = {}
    for direction in ("Demand", "Supply"):
        matching_zones = [zone for zone in active_zones if zone["direction"] == direction]
        latest_by_direction[direction] = matching_zones[-1] if matching_zones else None
    return zones, latest_by_direction
