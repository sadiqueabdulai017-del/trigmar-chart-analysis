"""Volume Profile calculations from observed quote-tick counts."""

from decimal import Decimal, ROUND_FLOOR
from math import isfinite


def build_volume_profile(tick_rows, bin_size, value_area_fraction=0.7):
    """Build a tick-activity estimate; it is not traded volume-at-price.

    The value area starts at the POC and expands through adjacent bins, adding
    the side with more activity until the requested fraction is reached. Equal
    adjacent counts expand toward the lower-price bin.
    """
    if not isfinite(bin_size) or bin_size <= 0:
        raise ValueError("bin_size must be a positive finite number.")
    if not isfinite(value_area_fraction) or not 0 < value_area_fraction <= 1:
        raise ValueError("value_area_fraction must be in (0, 1].")

    bins = {}
    ignored_rows = 0
    for row in tick_rows:
        try:
            price = float(row["price"])
            tick_count = int(row.get("tick_count", 1))
        except (KeyError, TypeError, ValueError, OverflowError):
            ignored_rows += 1
            continue
        if not isfinite(price) or tick_count <= 0:
            ignored_rows += 1
            continue
        bin_index = int(
            (Decimal(str(price)) / Decimal(str(bin_size))).to_integral_value(
                rounding=ROUND_FLOOR
            )
        )
        bins[bin_index] = bins.get(bin_index, 0) + tick_count

    if not bins:
        return None

    indexes = list(range(min(bins), max(bins) + 1))
    counts = [bins.get(index, 0) for index in indexes]
    total_ticks = sum(counts)
    poc_position = max(range(len(counts)), key=lambda index: (counts[index], -index))

    included_ticks = counts[poc_position]
    lower = upper = poc_position
    target_ticks = total_ticks * value_area_fraction
    while included_ticks < target_ticks and (lower > 0 or upper < len(counts) - 1):
        below = counts[lower - 1] if lower > 0 else -1
        above = counts[upper + 1] if upper < len(counts) - 1 else -1
        if below >= above:
            lower -= 1
            included_ticks += counts[lower]
        else:
            upper += 1
            included_ticks += counts[upper]

    nodes = []
    position = 0
    while position < len(counts):
        end_position = position + 1
        while (
            end_position < len(counts)
            and counts[end_position] == counts[position]
        ):
            end_position += 1
        if position > 0 and end_position < len(counts):
            count = counts[position]
            previous_count = counts[position - 1]
            next_count = counts[end_position]
            node_position = (position + end_position - 1) // 2
            if count > 0 and count > previous_count and count > next_count:
                nodes.append({
                    "type": "HVN",
                    "price": (indexes[node_position] + 0.5) * bin_size,
                    "tick_count": count,
                })
            elif count < previous_count and count < next_count:
                nodes.append({
                    "type": "LVN",
                    "price": (indexes[node_position] + 0.5) * bin_size,
                    "tick_count": count,
                })
        position = end_position

    profile_bins = [
        {
            "price": (index + 0.5) * bin_size,
            "tick_count": count,
        }
        for index, count in zip(indexes, counts)
    ]
    return {
        "bin_size": bin_size,
        "poc": (indexes[poc_position] + 0.5) * bin_size,
        "vah": (indexes[upper] + 1) * bin_size,
        "val": indexes[lower] * bin_size,
        "total_ticks": total_ticks,
        "value_area_fraction": value_area_fraction,
        "bins": profile_bins,
        "nodes": nodes,
        "ignored_rows": ignored_rows,
    }
