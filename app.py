import os
import streamlit as st
import json
import logging
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sqlite3
import threading
import time
import websocket
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from volume_profile import build_volume_profile
from backtesting import (
    count_same_day_time_gaps,
    evaluate_backtest,
    filter_combined_setups,
    summarize_trade_journal,
)
from supply_demand import calculate_supply_demand_zones

# -----------------------------------------------------------------------------
# 1. PAGE LAYOUT & GLOBAL STYLE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TrigMAR Trading Dashboard", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    div[data-testid="stMetricValue"] { color: #ffffff; font-size: 24px; font-weight: bold; }
    div[data-testid="stMetricLabel"] { color: #808495; }
    </style>
    """, unsafe_allow_html=True)

TIMEFRAME_SECONDS = {
    "1 Minute": 60,
    "5 Minute": 300,
    "15 Minute": 900,
    "1 Hour": 3600,
}
LIVE_HISTORY_WEEKDAYS = 30
CHART_HISTORY_WEEKDAYS = 5
HISTORY_PAGE_SIZE = 1000
DERIV_PUBLIC_WS_URL = "wss://api.derivws.com/trading/v1/options/ws/public"
CANDLE_DB_PATH = Path(__file__).resolve().parent / "data" / "candles.db"
logger = logging.getLogger(__name__)

APP_USERNAME = os.getenv("TRIGMAR_USERNAME", "")
APP_PASSWORD = os.getenv("TRIGMAR_PASSWORD", "")


def require_private_login():
    """Lock the dashboard behind a single-user login gate."""
    if not APP_USERNAME or not APP_PASSWORD:
        st.warning(
            "Private dashboard: set TRIGMAR_USERNAME and TRIGMAR_PASSWORD in the hosting environment before deployment. "
            "Access is blocked until those values are configured."
        )
        st.stop()

    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if st.session_state["authenticated"]:
        return

    st.markdown("## Private access")
    st.info("This dashboard is restricted to a single authorized user.")
    with st.form("private-login", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if username == APP_USERNAME and password == APP_PASSWORD:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Invalid username or password.")
    st.stop()


@st.cache_resource
def get_candle_feed_runtime():
    return {"threads": {}, "tick_threads": {}, "lock": threading.Lock()}


def trading_window_start(timestamp, weekdays=LIVE_HISTORY_WEEKDAYS):
    """Return UTC midnight at the start of the latest requested weekdays."""
    current_day = datetime.fromtimestamp(timestamp, tz=timezone.utc).date()
    weekdays_seen = 0
    while weekdays_seen < weekdays:
        if current_day.weekday() < 5:
            weekdays_seen += 1
        if weekdays_seen < weekdays:
            current_day -= timedelta(days=1)
    return int(datetime.combine(
        current_day, datetime.min.time(), tzinfo=timezone.utc
    ).timestamp())


def open_candle_database():
    CANDLE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(CANDLE_DB_PATH, timeout=15)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("""
        CREATE TABLE IF NOT EXISTS candles (
            symbol TEXT NOT NULL,
            granularity INTEGER NOT NULL,
            open_time INTEGER NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            PRIMARY KEY (symbol, granularity, open_time)
        )
    """)
    connection.execute("""
        CREATE INDEX IF NOT EXISTS candles_by_time
        ON candles (symbol, granularity, open_time)
    """)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS tick_activity (
            symbol TEXT NOT NULL,
            epoch INTEGER NOT NULL,
            quote REAL NOT NULL,
            tick_count INTEGER NOT NULL,
            PRIMARY KEY (symbol, epoch, quote)
        )
    """)
    connection.execute("""
        CREATE INDEX IF NOT EXISTS tick_activity_by_time
        ON tick_activity (symbol, epoch)
    """)
    return connection


def save_tick_activity(symbol, tick):
    """Persist observed quote ticks as activity counts, not traded volume."""
    try:
        epoch = int(tick["epoch"])
        quote = float(tick["quote"])
    except (KeyError, TypeError, ValueError, OverflowError):
        logger.warning("Ignoring malformed quote tick for %s.", symbol)
        return
    if not np.isfinite(quote) or datetime.fromtimestamp(
        epoch, tz=timezone.utc
    ).weekday() >= 5:
        return

    with closing(open_candle_database()) as connection:
        with connection:
            connection.execute(
                """
                INSERT INTO tick_activity (symbol, epoch, quote, tick_count)
                VALUES (?, ?, ?, 1)
                ON CONFLICT (symbol, epoch, quote) DO UPDATE SET
                    tick_count = tick_count + 1
                """,
                (symbol, epoch, quote),
            )
            connection.execute(
                "DELETE FROM tick_activity WHERE epoch < ?",
                (trading_window_start(epoch),),
            )


def previous_utc_session(timestamp):
    """Return the previous weekday UTC session as [start, end)."""
    session_date = datetime.fromtimestamp(
        timestamp, tz=timezone.utc
    ).date() - timedelta(days=1)
    while session_date.weekday() >= 5:
        session_date -= timedelta(days=1)
    start = datetime.combine(
        session_date, datetime.min.time(), tzinfo=timezone.utc
    )
    return int(start.timestamp()), int((start + timedelta(days=1)).timestamp())


def load_previous_session_profile(symbol, timestamp, asset):
    start, end = previous_utc_session(timestamp)
    with closing(open_candle_database()) as connection:
        rows = [
            {"price": quote, "tick_count": tick_count}
            for quote, tick_count in connection.execute(
                """
                SELECT quote, SUM(tick_count)
                FROM tick_activity
                WHERE symbol = ? AND epoch >= ? AND epoch < ?
                GROUP BY quote
                ORDER BY quote
                """,
                (symbol, start, end),
            )
        ]
        coverage = connection.execute(
            """
            SELECT MIN(epoch), MAX(epoch), SUM(tick_count)
            FROM tick_activity
            WHERE symbol = ? AND epoch >= ? AND epoch < ?
            """,
            (symbol, start, end),
        ).fetchone()

    if not coverage or coverage[0] is None:
        return {
            "profile": None,
            "start": start,
            "end": end,
            "reason": "No quote-tick activity was collected for the previous UTC session.",
        }
    if coverage[0] > start + 300 or coverage[1] < end - 300:
        return {
            "profile": None,
            "start": start,
            "end": end,
            "reason": "Previous-session tick history is incomplete; levels are withheld.",
        }

    bin_size = 0.01 if asset == "Gold (XAUUSD)" else 0.0001
    profile = build_volume_profile(rows, bin_size)
    return {
        "profile": profile,
        "start": start,
        "end": end,
        "reason": None if profile else "No valid quote ticks were available.",
        "tick_count": int(coverage[2] or 0),
    }


def render_backtest_report(
    candles,
    baseline_setups,
    target_multiple,
    source_label,
    granularity_seconds,
    round_trip_cost_r,
    combined_setups=None,
    unavailable_profile_count=0,
):
    """Render historical setup results without treating them as forecasts."""
    split_index = int(len(candles) * 0.7)
    train_end = split_index if split_index > 0 else len(candles)
    periods = [("In-sample (first 70%)", 0, train_end)]
    if split_index > 0 and split_index < len(candles):
        periods.append(
            ("Out-of-sample (last 30%)", split_index, len(candles))
        )
    strategies = [("Existing breakout", baseline_setups)]
    if combined_setups is not None:
        strategies.append((
            "Combined (breakout + prior-session VA)",
            combined_setups,
        ))

    summaries = [
        (
            f"{strategy} · {period}",
            evaluate_backtest(
                candles,
                setups,
                start_index=start,
                end_index=end,
                round_trip_cost_r=round_trip_cost_r,
            ),
        )
        for strategy, setups in strategies
        for period, start, end in periods
        if end > start
    ]
    trade_journal = [
        trade
        for _, summary in summaries
        for trade in summary["trades"]
    ]
    mistake_summary = summarize_trade_journal(trade_journal)

    with st.expander("Historical Backtest", expanded=False):
        history_range = (
            f"{pd.Timestamp(candles['Time'].iloc[0]).strftime('%Y-%m-%d %H:%M')} "
            f"to {pd.Timestamp(candles['Time'].iloc[-1]).strftime('%Y-%m-%d %H:%M')}"
            if not candles.empty and "Time" in candles
            else "no completed candle history"
        )
        st.caption(
            f"Source: {source_label} · {history_range} · "
            f"{len(candles):,} closed candles · "
            f"70/30 chronological split · target {target_multiple:g}R · "
            f"assumed round-trip costs {round_trip_cost_r:.2f}R per completed trade."
        )
        gap_count = count_same_day_time_gaps(candles, granularity_seconds)
        st.caption(
            f"Observed same-UTC-day timestamp gaps: {gap_count}. This counts "
            "adjacent candle pairs with a gap, not the number of missing candles."
        )
        if gap_count:
            st.warning(
                "The backtest data contains same-day candle gaps. Results may be "
                "less reliable across those periods; inspect the source history."
            )
        report_rows = []
        for label, summary in summaries:
            report_rows.append({
                "Strategy / period": label,
                "Completed": summary["completed"],
                "Open": summary["open"],
                "Skipped overlap": summary["skipped_overlapping"],
                "Gross wins": summary["wins"],
                "Gross losses": summary["losses"],
                "Gross win rate": (
                    f"{summary['win_rate']:.1%}"
                    if summary["win_rate"] is not None else "N/A"
                ),
                "Net win rate": (
                    f"{summary['net_win_rate']:.1%}"
                    if summary["net_win_rate"] is not None else "N/A"
                ),
                "Net expectancy (R)": (
                    f"{summary['expectancy_r']:+.2f}"
                    if summary["expectancy_r"] is not None else "N/A"
                ),
                "Profit factor": (
                    f"{summary['profit_factor']:.2f}"
                    if summary["profit_factor"] is not None else "N/A"
                ),
                "Max drawdown (R)": f"{summary['max_drawdown_r']:.2f}",
                "Ambiguous bars": summary["ambiguous"],
            })
        st.dataframe(
            pd.DataFrame(report_rows),
            hide_index=True,
            width="stretch",
        )
        if any(summary["sample_warning"] for _, summary in summaries):
            st.warning(
                "Small sample: fewer than 30 completed trades in at least one "
                "strategy/period. Performance metrics may be unstable."
            )
        if mistake_summary["stops_hit"] or mistake_summary["targets_hit"]:
            st.info(
                "Mistake learning: "
                f"{mistake_summary['stops_hit']} stop losses hit, "
                f"{mistake_summary['targets_hit']} targets hit, "
                f"dominant mistake is {mistake_summary['most_common_mistake']} "
                f"({mistake_summary['pattern']})"
            )
        if unavailable_profile_count:
            st.info(
                f"{unavailable_profile_count} baseline setup(s) were excluded "
                "from Combined because the preceding session lacked complete "
                "observed tick activity. No volume levels were fabricated."
            )
        st.caption(
            "A candle touching both stop and target is counted as a loss because "
            "OHLC data cannot determine which was hit first. Open trades are "
            "excluded from win rate and expectancy. This report models one open "
            "trade at a time; later signals during an open trade are skipped. "
            "Costs are user-entered R assumptions, not measured broker spreads "
            "or commissions. Out-of-sample trades are evaluated only inside the "
            "final chronological 30% window; unresolved trades at its end stay open. "
            "Profile-only mode has no standalone entry rules, so it is not treated "
            "as a trading strategy. The system also records recurring stop-loss "
            "mistakes so the dashboard learns which direction is failing most often."
        )


def save_candle_rows(symbol, granularity, rows, market_time):
    """Upsert Deriv history and discard rows outside the retained weekday window."""
    cutoff = trading_window_start(market_time)
    prepared_rows = []
    for row in rows:
        raw_open_time = row.get("open_time")
        if raw_open_time is None:
            raw_open_time = row.get("epoch")
        if raw_open_time is None:
            continue
        open_time = int(raw_open_time)
        if open_time % granularity != 0:
            continue
        if datetime.fromtimestamp(open_time, tz=timezone.utc).weekday() >= 5:
            continue
        prepared_rows.append((
            symbol,
            granularity,
            open_time,
            float(row["open"]),
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
        ))

    with closing(open_candle_database()) as connection:
        with connection:
            connection.executemany(
                """
                INSERT INTO candles (
                    symbol, granularity, open_time, open, high, low, close
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (symbol, granularity, open_time) DO UPDATE SET
                    open = excluded.open,
                    high = excluded.high,
                    low = excluded.low,
                    close = excluded.close
                """,
                prepared_rows,
            )
            connection.execute(
                """
                DELETE FROM candles
                WHERE symbol = ? AND granularity = ? AND open_time < ?
                """,
                (symbol, granularity, cutoff),
            )


def load_candle_rows(symbol, granularity):
    with closing(open_candle_database()) as connection:
        with connection:
            connection.execute(
                """
                DELETE FROM candles
                WHERE symbol = ? AND granularity = ? AND open_time % ? != 0
                """,
                (symbol, granularity, granularity),
            )
        return [
            {
                "epoch": open_time,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
            }
            for open_time, open_price, high, low, close in connection.execute(
                """
                SELECT open_time, open, high, low, close
                FROM candles
                WHERE symbol = ? AND granularity = ?
                ORDER BY open_time
                """,
                (symbol, granularity),
            )
        ]


def candle_feed_worker(symbol, granularity):
    """Persist Deriv history and live candle updates independently of the browser."""
    while True:
        connection = None
        try:
            connection = websocket.create_connection(
                DERIV_PUBLIC_WS_URL, timeout=15
            )
            connection.send(json.dumps({"time": 1, "req_id": 1}))
            connection.settimeout(10)
            server_time_response = json.loads(connection.recv())
            if server_time_response.get("error"):
                raise RuntimeError(server_time_response["error"]["message"])
            if not server_time_response.get("time"):
                raise RuntimeError("Deriv did not return market server time.")
            market_time = int(server_time_response["time"])
            window_start = trading_window_start(market_time)
            request_id = 2
            request_end = "latest"
            previous_oldest = None
            while True:
                history_request = {
                    "ticks_history": symbol,
                    "start": window_start,
                    "end": request_end,
                    "granularity": granularity,
                    "count": HISTORY_PAGE_SIZE,
                    "style": "candles",
                    "req_id": request_id,
                }
                connection.send(json.dumps(history_request))
                while True:
                    history_response = json.loads(connection.recv())
                    if int(history_response.get("req_id", 0)) == request_id:
                        break
                    if history_response.get("error"):
                        raise RuntimeError(
                            history_response["error"]["message"]
                        )
                if history_response.get("error"):
                    raise RuntimeError(history_response["error"]["message"])
                rows = history_response.get("candles", [])
                if not rows:
                    break
                save_candle_rows(symbol, granularity, rows, market_time)
                oldest = min(int(row["epoch"]) for row in rows)
                if oldest <= window_start or oldest == previous_oldest:
                    if oldest > window_start:
                        logger.warning(
                            "Historical candle paging stopped before the "
                            "retained weekday window for %s at %ss.",
                            symbol,
                            granularity,
                        )
                    break
                previous_oldest = oldest
                request_end = oldest - 1
                request_id += 1

            request_id += 1
            connection.send(json.dumps({
                "ticks_history": symbol,
                "end": "latest",
                "granularity": granularity,
                "count": 1,
                "style": "candles",
                "subscribe": 1,
                "req_id": request_id,
            }))
            while True:
                subscription_response = json.loads(connection.recv())
                if int(subscription_response.get("req_id", 0)) == request_id:
                    break
                if subscription_response.get("error"):
                    raise RuntimeError(subscription_response["error"]["message"])
            if subscription_response.get("error"):
                raise RuntimeError(subscription_response["error"]["message"])
            rows = subscription_response.get("candles", [])
            if rows:
                save_candle_rows(symbol, granularity, rows, market_time)
            connection.settimeout(5)
            last_time_sync = time.monotonic()
            while True:
                if time.monotonic() - last_time_sync >= 60:
                    connection.send(json.dumps({"time": 1, "req_id": 1}))
                    last_time_sync = time.monotonic()
                try:
                    message = json.loads(connection.recv())
                except websocket.WebSocketTimeoutException:
                    continue
                if int(message.get("req_id", 0)) == 1:
                    if message.get("error"):
                        raise RuntimeError(message["error"]["message"])
                    if message.get("time") is None:
                        logger.warning(
                            "Deriv time sync response omitted server time for "
                            "%s at %ss; retaining the previous time.",
                            symbol,
                            granularity,
                        )
                        continue
                    market_time = int(message["time"])
                    continue
                if message.get("error"):
                    raise RuntimeError(message["error"]["message"])
                rows = message.get("candles") or (
                    [message["ohlc"]] if message.get("ohlc") else []
                )
                if rows:
                    save_candle_rows(symbol, granularity, rows, market_time)
        except Exception:
            logger.exception(
                "Live candle persistence failed for %s at %ss; retrying.",
                symbol,
                granularity,
            )
            time.sleep(5)
        finally:
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    logger.exception(
                        "Failed to close live candle feed for %s at %ss.",
                        symbol,
                        granularity,
                    )


def ensure_candle_feed(symbol, granularity):
    key = (symbol, granularity)
    runtime = get_candle_feed_runtime()
    with runtime["lock"]:
        thread = runtime["threads"].get(key)
        if thread is None or not thread.is_alive():
            thread = threading.Thread(
                target=candle_feed_worker,
                args=key,
                name=f"candle-feed-{symbol}-{granularity}",
                daemon=True,
            )
            runtime["threads"][key] = thread
            thread.start()


def tick_feed_worker(symbol):
    """Persist live quote updates once per instrument, independent of candle size."""
    while True:
        connection = None
        try:
            connection = websocket.create_connection(
                DERIV_PUBLIC_WS_URL, timeout=15
            )
            connection.send(json.dumps({
                "ticks": symbol,
                "subscribe": 1,
                "req_id": 1,
            }))
            connection.settimeout(10)
            while True:
                try:
                    message = json.loads(connection.recv())
                except websocket.WebSocketTimeoutException:
                    continue
                if message.get("error"):
                    raise RuntimeError(message["error"]["message"])
                tick = message.get("tick")
                if tick:
                    save_tick_activity(symbol, tick)
        except Exception:
            logger.exception(
                "Live tick-activity persistence failed for %s; retrying.",
                symbol,
            )
            time.sleep(5)
        finally:
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    logger.exception(
                        "Failed to close live tick feed for %s.",
                        symbol,
                    )


def ensure_tick_feed(symbol):
    runtime = get_candle_feed_runtime()
    with runtime["lock"]:
        thread = runtime["tick_threads"].get(symbol)
        if thread is None or not thread.is_alive():
            thread = threading.Thread(
                target=tick_feed_worker,
                args=(symbol,),
                name=f"tick-feed-{symbol}",
                daemon=True,
            )
            runtime["tick_threads"][symbol] = thread
            thread.start()

# -----------------------------------------------------------------------------
# 2. LOCAL SIMULATOR ENGINE
# -----------------------------------------------------------------------------
def generate_simulated_pattern(
    interval_minutes=5,
    asset="Gold (XAUUSD)",
):
    """Generate a balanced price series with both bullish and bearish breakouts."""
    rng = np.random.default_rng(42)
    interval_seconds = interval_minutes * 60
    candle_count = 160 if interval_minutes == 15 else 60
    last_closed_open = (
        int(datetime.now().timestamp()) // interval_seconds - 1
    ) * interval_seconds
    base_time = datetime.fromtimestamp(
        last_closed_open - interval_seconds * (candle_count - 1)
    )
    times = [
        base_time + timedelta(minutes=interval_minutes * i)
        for i in range(candle_count)
    ]

    base_price, volatility = {
        "Gold (XAUUSD)": (2000.0, 1.4),
        "EURUSD": (1.080, 0.0014),
        "GBPUSD": (1.270, 0.0018),
    }[asset]

    closes = np.empty(candle_count)
    opens = np.empty(candle_count)
    closes[0] = base_price
    opens[0] = base_price
    first_turn = candle_count // 3
    second_turn = 2 * candle_count // 3
    for index in range(1, candle_count):
        drift = 0.0
        if first_turn <= index < second_turn:
            drift = -0.35 * volatility
        elif index >= second_turn:
            drift = 0.45 * volatility
        opens[index] = closes[index - 1]
        closes[index] = opens[index] + rng.normal(drift, volatility * 0.35)

    upper_wicks = rng.uniform(0.1, 0.45, candle_count) * volatility
    lower_wicks = rng.uniform(0.1, 0.45, candle_count) * volatility

    df = pd.DataFrame({
        "Time": times,
        "Open": opens,
        "High": np.maximum(opens, closes) + upper_wicks,
        "Low": np.minimum(opens, closes) - lower_wicks,
        "Close": closes,
    })
    return df

# -----------------------------------------------------------------------------
# 3. LIVE BROWSER CHART
# -----------------------------------------------------------------------------
def build_live_chart_html(
    symbol,
    granularity_seconds,
    asset,
    analysis_mode,
    volume_profile,
    target_multiple,
    starting_balance,
    risk_percentage,
    initial_candles,
):
    """Build a browser-owned chart that updates without Streamlit reruns."""
    config = json.dumps({
        "symbol": symbol,
        "granularity": granularity_seconds,
        "asset": asset,
        "analysisMode": analysis_mode,
        "volumeProfile": volume_profile,
        "targetMultiple": target_multiple,
        "riskBudget": starting_balance * risk_percentage / 100,
        "initialCandles": initial_candles,
    })
    return r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
* { box-sizing: border-box; }
body { margin: 0; background: #0e1117; color: #e8eaf0; font: 14px Arial, sans-serif; }
.wrap { width: 100%; }
.status { color: #aeb4c2; margin: 4px 2px 10px; min-height: 18px; }
.toolbar { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; margin-bottom: 8px; }
button { color: #d8dce5; background: #222630; border: 1px solid #414755; border-radius: 4px; padding: 7px 12px; cursor: pointer; }
button.active { color: #fff; background: #1769aa; border-color: #42a5f5; }
.hint { color: #8991a1; margin-left: 8px; }
canvas { display: block; width: 100%; height: 510px; background: #0e1117; border: 1px solid #252a34; touch-action: none; }
.ruler { display: flex; align-items: center; gap: 18px; flex-wrap: wrap; margin: 12px 0; }
.ruler label { color: #aeb4c2; display: flex; gap: 8px; align-items: center; }
input[type=range] { width: 150px; accent-color: #fbc02d; }
.metrics { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 8px; }
.metric { background: #171b23; border: 1px solid #292f3a; border-radius: 6px; padding: 10px 14px; min-width: 145px; }
.label { color: #8991a1; font-size: 12px; margin-bottom: 5px; }
.value { color: #fff; font-size: 17px; font-weight: 700; }
.signal { color: #aeb4c2; margin: 10px 2px; }
.history { max-height: 190px; overflow: auto; margin-top: 10px; }
.history table { width: 100%; border-collapse: collapse; font-size: 12px; }
.history th, .history td { padding: 6px 8px; border-bottom: 1px solid #292f3a; text-align: left; white-space: nowrap; }
.history th { color: #aeb4c2; position: sticky; top: 0; background: #0e1117; }
.error { color: #ff7b7b; }
</style>
</head>
<body>
<div class="wrap">
  <div id="status" class="status">Connecting to Deriv public feed…</div>
  <div class="toolbar">
    <button id="pan" class="active">Pan</button>
    <button id="zoom">Zoom</button>
    <button id="zoomIn">+</button>
    <button id="zoomOut">−</button>
    <button id="reset">Reset view</button>
    <span class="hint">Pan is active by default. Drag to move; use the wheel to zoom.</span>
  </div>
  <canvas id="chart" aria-label="Live candlestick chart"></canvas>
  <div class="ruler">
    <label>Ruler start <input id="start" type="range" min="0" max="0" value="0"><span id="startValue">—</span></label>
    <label>Ruler end <input id="end" type="range" min="0" max="0" value="0"><span id="endValue">—</span></label>
  </div>
  <div class="metrics">
    <div class="metric"><div class="label">Automatic signal</div><div id="direction" class="value">Scanning…</div></div>
    <div class="metric"><div class="label">Entry</div><div id="entry" class="value">—</div></div>
    <div class="metric"><div class="label">Stop loss</div><div id="stop" class="value">—</div></div>
    <div class="metric"><div class="label">Take profit</div><div id="target" class="value">—</div></div>
    <div class="metric"><div class="label">Ruler price change</div><div id="change" class="value">—</div></div>
  </div>
  <div id="signal" class="signal">Signals are informational only. This app does not place trades.</div>
  <div id="structure" class="signal">Market structure: waiting for confirmed swings.</div>
  <div id="liquidity" class="signal">Liquidity sweeps: waiting for closed candles.</div>
  <div id="zones" class="signal">Supply/demand: scanning closed candles.</div>
  <div id="sneaky" class="signal">Sneaky Pivot: 15-minute setup only; waiting for previous-day levels.</div>
  <div id="history" class="history"></div>
</div>
<script>
const config = __LIVE_CONFIG__;
const candles = [];
const previousDayCandles = [];
const setupHistory = [];
const canvas = document.getElementById("chart");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const startSlider = document.getElementById("start");
const endSlider = document.getElementById("end");
let visibleCount = 60;
let viewEnd = 59;
let initializedView = false;
let mode = "pan";
let dragStart = null;
let initializedRuler = false;
let ws = null;
let retryTimer = null;
let timeSyncTimer = null;
let serverTimeOffsetSeconds = null;
let dailySubscriptionId = null;
let historyRequestId = null;
let nextHistoryRequestId = 100;
let historyWindowStart = null;
let previousHistoryOldest = null;
let dailyFeedError = "";
let feedClockError = "";
let renderScheduled = false;
const HISTORY_PAGE_SIZE = 1000;

const fmt = value => Number(value).toLocaleString(undefined, {
  minimumFractionDigits: config.asset === "Gold (XAUUSD)" ? 2 : 5,
  maximumFractionDigits: config.asset === "Gold (XAUUSD)" ? 2 : 5
});
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
function feedNowSeconds() {
  return serverTimeOffsetSeconds === null
    ? NaN
    : Date.now() / 1000 + serverTimeOffsetSeconds;
}
function setStatus(text, error=false) {
  statusEl.textContent = text;
  statusEl.className = error ? "status error" : "status";
}
function fromCandle(row) {
  return {
    time: Number(row.open_time ?? row.epoch),
    open: Number(row.open),
    high: Number(row.high),
    low: Number(row.low),
    close: Number(row.close)
  };
}
function tradingWindowStart(nowSeconds) {
  if (!Number.isFinite(nowSeconds)) return NaN;
  const current = new Date(nowSeconds * 1000);
  let day = Date.UTC(current.getUTCFullYear(), current.getUTCMonth(), current.getUTCDate());
  let weekdaysSeen = 0;
  while (weekdaysSeen < 5) {
    const weekday = new Date(day).getUTCDay();
    if (weekday >= 1 && weekday <= 5) weekdaysSeen++;
    if (weekdaysSeen < 5) day -= 86400 * 1000;
  }
  return day / 1000;
}
function isWeekdayCandle(candle) {
  const weekday = new Date(candle.time * 1000).getUTCDay();
  return weekday >= 1 && weekday <= 5;
}
function mergePreviousDayRows(rows) {
  for (const row of rows) {
    const candle = fromCandle(row);
    const index = previousDayCandles.findIndex(item => item.time === candle.time);
    if (index >= 0) previousDayCandles[index] = candle;
    else previousDayCandles.push(candle);
  }
  previousDayCandles.sort((a, b) => a.time - b.time);
  if (previousDayCandles.length > 4) previousDayCandles.splice(0, previousDayCandles.length - 4);
  dailyFeedError = "";
  render();
}
function previousDayRange() {
  const completed = previousDayCandles.filter(
    candle => candle.time + 86400 <= feedNowSeconds()
  );
  return completed.length ? completed[completed.length - 1] : null;
}
function mergeRows(rows) {
  const followLatest = initializedView && viewEnd >= candles.length - 1;
  for (const row of rows) {
    const candle = fromCandle(row);
    if (![candle.time, candle.open, candle.high, candle.low, candle.close].every(Number.isFinite)) {
      setStatus("Received invalid candle data from market feed.", true);
      continue;
    }
    if (candle.time % config.granularity !== 0) continue;
    if (!isWeekdayCandle(candle)) continue;
    const index = candles.findIndex(item => item.time === candle.time);
    if (index >= 0) candles[index] = candle;
    else candles.push(candle);
  }
  candles.sort((a, b) => a.time - b.time);
  const nowSeconds = feedNowSeconds();
  const cutoff = Number.isFinite(nowSeconds)
    ? tradingWindowStart(nowSeconds)
    : NaN;
  if (Number.isFinite(cutoff)) {
    const firstRetained = candles.findIndex(candle => candle.time >= cutoff);
    if (firstRetained > 0) candles.splice(0, firstRetained);
    else if (firstRetained === -1) candles.splice(0, candles.length);
  }
  if (!initializedView && candles.length) {
    viewEnd = candles.length - 1;
    visibleCount = Math.min(60, candles.length);
    initializedView = true;
  } else if (followLatest && candles.length) {
    viewEnd = candles.length - 1;
  }
  visibleCount = clamp(visibleCount, 8, Math.max(60, candles.length));
  viewEnd = clamp(viewEnd, Math.min(visibleCount - 1, candles.length - 1), candles.length - 1);
  updateSetupHistory();
  updateRulerControls();
  render();
}
function signalsForSeries() {
  const signals = [];
  if (serverTimeOffsetSeconds === null || config.analysisMode === "Volume Profile") return signals;
  const lastByDirection = {BUY: -20, SELL: -20};
  const now = feedNowSeconds();
  for (let i = 21; i < candles.length; i++) {
    const reference = candles.slice(i - 21, i - 1);
    const previous = candles.slice(i - 20, i);
    const current = candles[i];
    if (current.time + config.granularity > now) continue;
    const resistance = Math.max(...reference.map(item => item.high));
    const support = Math.min(...reference.map(item => item.low));
    const prevClose = candles[i - 1].close;
    let direction, stop, risk, targetDirection;
    if (prevClose <= resistance && current.close > resistance && current.close > current.open) {
      direction = "BUY";
      stop = Math.min(...previous.slice(-10).map(item => item.low));
      risk = current.close - stop;
      targetDirection = 1;
    } else if (prevClose >= support && current.close < support && current.close < current.open) {
      direction = "SELL";
      stop = Math.max(...previous.slice(-10).map(item => item.high));
      risk = stop - current.close;
      targetDirection = -1;
    } else continue;
    if (config.analysisMode === "Combined") {
      const profile = config.volumeProfile;
      if (!profile) continue;
      if (direction === "BUY" && current.close <= profile.vah) continue;
      if (direction === "SELL" && current.close >= profile.val) continue;
    }
    if (risk <= 0 || i - lastByDirection[direction] < 20) continue;
    lastByDirection[direction] = i;
    signals.push({
      id: `${current.time}:${direction}`,
      index: i,
      candleTime: current.time,
      direction,
      entry: current.close,
      stop,
      target: current.close + targetDirection * risk * config.targetMultiple,
      signalTime: current.time + config.granularity,
      outcome: "No level reached yet"
    });
  }
  return signals;
}
function marketStructureForSeries() {
  const closedCount = candles.filter(
    candle => candle.time + config.granularity <= feedNowSeconds()
  ).length;
  const swings = [];
  const breaks = [];
  const sweeps = [];
  const highs = [];
  const lows = [];
  let bias = "Neutral";
  let lastSwingHigh = null;
  let lastSwingLow = null;
  let brokenHighTime = null;
  let brokenLowTime = null;
  const sweptHighTimes = new Set();
  const sweptLowTimes = new Set();
  for (let index = 1; index < closedCount; index++) {
    const current = candles[index];
    const previous = candles[index - 1];
    if (lastSwingHigh && brokenHighTime !== lastSwingHigh.time
        && previous.close <= lastSwingHigh.price && current.close > lastSwingHigh.price) {
      const event = {index, label: bias === "Bearish" ? "CHoCH ↑" : "BOS ↑"};
      breaks.push(event);
      bias = "Bullish";
      brokenHighTime = lastSwingHigh.time;
    }
    if (lastSwingLow && brokenLowTime !== lastSwingLow.time
        && previous.close >= lastSwingLow.price && current.close < lastSwingLow.price) {
      const event = {index, label: bias === "Bullish" ? "CHoCH ↓" : "BOS ↓"};
      breaks.push(event);
      bias = "Bearish";
      brokenLowTime = lastSwingLow.time;
    }
    if (lastSwingLow && !sweptLowTimes.has(lastSwingLow.time)
        && current.low < lastSwingLow.price && current.close > lastSwingLow.price) {
      sweeps.push({
        index, direction: "Bullish", label: "SSL sweep ↑",
        price: current.low, level: lastSwingLow.price
      });
      sweptLowTimes.add(lastSwingLow.time);
    }
    if (lastSwingHigh && !sweptHighTimes.has(lastSwingHigh.time)
        && current.high > lastSwingHigh.price && current.close < lastSwingHigh.price) {
      sweeps.push({
        index, direction: "Bearish", label: "BSL sweep ↓",
        price: current.high, level: lastSwingHigh.price
      });
      sweptHighTimes.add(lastSwingHigh.time);
    }
    const pivotIndex = index - 2;
    if (pivotIndex < 2) continue;
    const window = candles.slice(pivotIndex - 2, index + 1);
    const pivot = candles[pivotIndex];
    const highMax = Math.max(...window.map(item => item.high));
    const lowMin = Math.min(...window.map(item => item.low));
    let newSwing = false;
    if (pivot.high === highMax && window.filter(item => item.high === highMax).length === 1) {
      const previousHigh = highs[highs.length - 1];
      const label = previousHigh ? (pivot.high > previousHigh.price ? "HH" : "LH") : "SH";
      const swing = {index: pivotIndex, time: pivot.time, price: pivot.high, label};
      highs.push(swing);
      swings.push(swing);
      lastSwingHigh = swing;
      newSwing = true;
    }
    if (pivot.low === lowMin && window.filter(item => item.low === lowMin).length === 1) {
      const previousLow = lows[lows.length - 1];
      const label = previousLow ? (pivot.low > previousLow.price ? "HL" : "LL") : "SL";
      const swing = {index: pivotIndex, time: pivot.time, price: pivot.low, label};
      lows.push(swing);
      swings.push(swing);
      lastSwingLow = swing;
      newSwing = true;
    }
    if (newSwing) {
      const latestHighLabel = highs.length ? highs[highs.length - 1].label : "";
      const latestLowLabel = lows.length ? lows[lows.length - 1].label : "";
      if (latestHighLabel === "HH" && latestLowLabel === "HL") bias = "Bullish";
      else if (latestHighLabel === "LH" && latestLowLabel === "LL") bias = "Bearish";
    }
  }
  return {swings, breaks, sweeps, bias};
}
function supplyDemandForSeries() {
  const closedCount = candles.filter(
    candle => candle.time + config.granularity <= feedNowSeconds()
  ).length;
  const zones = [];
  const averageRangePeriod = 10;
  const impulseLength = 3;
  for (let end = averageRangePeriod + impulseLength; end < closedCount; end++) {
    const originIndex = end - impulseLength;
    const origin = candles[originIndex];
    const base = candles.slice(originIndex - averageRangePeriod, originIndex);
    const impulse = candles.slice(originIndex + 1, end + 1);
    if (impulse.some(candle => candle.high <= candle.low)) continue;
    const averageRange = base.reduce(
      (total, candle) => total + candle.high - candle.low, 0
    ) / averageRangePeriod;
    const bullish = impulse.every(candle =>
      candle.close > candle.open
      && candle.high > candle.low
      && (candle.close - candle.open) / (candle.high - candle.low) >= 0.6
    );
    const bearish = impulse.every(candle =>
      candle.close < candle.open
      && candle.high > candle.low
      && (candle.open - candle.close) / (candle.high - candle.low) >= 0.6
    );
    const displacement = impulse[impulse.length - 1].close - origin.open;
    const displacementRatio = Math.abs(displacement) / averageRange;
    const bodyEfficiency = impulse.reduce(
      (total, candle) => total + Math.abs(candle.close - candle.open) /
        (candle.high - candle.low), 0
    ) / impulseLength;
    const strengthScore = Math.round(
      60 * Math.min(displacementRatio / 3, 1)
      + 40 * Math.min(bodyEfficiency / 0.9, 1)
    );
    const strength = strengthScore >= 80
      ? "Strong" : strengthScore >= 65 ? "Moderate" : "Developing";
    const brokeStructureUp = impulse[impulse.length - 1].close
      > Math.max(...base.map(candle => candle.high));
    const brokeStructureDown = impulse[impulse.length - 1].close
      < Math.min(...base.map(candle => candle.low));
    let direction = null;
    if (bullish && origin.close < origin.open
        && displacement >= averageRange * 1.5 && brokeStructureUp) {
      direction = "Demand";
    } else if (
      bearish && origin.close > origin.open
      && -displacement >= averageRange * 1.5 && brokeStructureDown
    ) {
      direction = "Supply";
    }
    if (!direction) continue;
    const zone = {
      direction,
      originIndex,
      startTime: origin.time,
      low: origin.low,
      high: origin.high,
      impulseEnd: end,
      strengthScore,
      strength,
      status: "Active",
      retestIndex: null,
      touches: 0
    };
    let touching = false;
    for (let index = end + 1; index < closedCount; index++) {
      const candle = candles[index];
      const invalidated = direction === "Demand"
        ? candle.close < zone.low : candle.close > zone.high;
      if (invalidated) {
        zone.status = "Invalidated";
        break;
      }
      const touched = candle.low <= zone.high && candle.high >= zone.low;
      if (touched && !touching) zone.touches++;
      touching = touched;
      if (touched && zone.status === "Active") zone.status = "Tested";
      const confirmed = direction === "Demand"
        ? touched && candle.close > zone.high && candle.close > candle.open
        : touched && candle.close < zone.low && candle.close < candle.open;
      if (confirmed && zone.status !== "Retest confirmed") {
        zone.status = "Retest confirmed";
        zone.retestIndex = index;
      }
    }
    zones.push(zone);
  }
  const active = zones.filter(zone => zone.status !== "Invalidated");
  const demands = active.filter(zone => zone.direction === "Demand");
  const supplies = active.filter(zone => zone.direction === "Supply");
  const latestDemand = demands.length ? demands[demands.length - 1] : null;
  const latestSupply = supplies.length ? supplies[supplies.length - 1] : null;
  return {zones, latestDemand, latestSupply};
}
function sneakyPivotForSeries(structure) {
  const daily = previousDayRange();
  if (serverTimeOffsetSeconds === null) {
    return {setups: [], daily, reason: "Synchronizing with market server time."};
  }
  if (config.granularity !== 900) {
    return {setups: [], daily, reason: "Available on the 15-minute chart only."};
  }
  if (!daily) {
    const reason = dailyFeedError
      ? `Previous-day levels unavailable: ${dailyFeedError}`
      : "Waiting for previous-day candle data.";
    return {setups: [], daily: null, reason};
  }
  const closed = candles.filter(
    candle => candle.time + config.granularity <= feedNowSeconds()
  );
  const setups = [];
  let pending = null;
  for (let index = 1; index < closed.length; index++) {
    const current = closed[index];
    const previous = closed[index - 1];
    if (pending) {
      if (!pending.confirmed) {
        const confirmation = pending.direction === "SELL"
          ? current.close < current.open && current.close < pending.level
          : current.close > current.open && current.close > pending.level;
        if (confirmation) {
          pending.confirmed = true;
          pending.confirmationIndex = index;
        } else pending = null;
        continue;
      }
      if (index <= pending.confirmationIndex) continue;
      const invalidated = pending.direction === "SELL"
        ? current.close > pending.sneakyHigh : current.close < pending.sneakyLow;
      if (invalidated) {
        pending = null;
        continue;
      }
      const broken = pending.direction === "SELL"
        ? previous.close >= pending.sneakyLow && current.close < pending.sneakyLow
        : previous.close <= pending.sneakyHigh && current.close > pending.sneakyHigh;
      if (!broken) continue;
      const stop = pending.direction === "SELL" ? pending.sneakyHigh : pending.sneakyLow;
      const risk = Math.abs(current.close - stop);
      if (risk > 0) {
        const targetDirection = pending.direction === "SELL" ? -1 : 1;
        setups.push({
          id: `${current.time}:${pending.direction}:sneaky`,
          direction: pending.direction,
          levelName: pending.levelName,
          sweptLevel: pending.level,
          sneakyIndex: pending.sneakyIndex,
          index,
          candleTime: current.time,
          signalTime: current.time + config.granularity,
          entry: current.close,
          stop,
          target: current.close + targetDirection * risk * config.targetMultiple
        });
      }
      pending = null;
      continue;
    }

    const priorSwings = structure.swings.filter(swing => swing.index + 2 < index);
    const latestHigh = priorSwings.filter(swing =>
      swing.label==="HH"||swing.label==="LH"||swing.label==="SH"
    ).slice(-1)[0];
    const latestLow = priorSwings.filter(swing =>
      swing.label==="HL"||swing.label==="LL"||swing.label==="SL"
    ).slice(-1)[0];
    const levels = [
      {name:"Previous-day high", type:"high", price:daily.high},
      {name:"Previous-day low", type:"low", price:daily.low}
    ];
    if (latestHigh) levels.push({name:latestHigh.label, type:"high", price:latestHigh.price});
    if (latestLow) levels.push({name:latestLow.label, type:"low", price:latestLow.price});
    const candidate = levels.find(level => level.type==="high"
      ? current.high > level.price && current.close < level.price && current.close < current.open
      : current.low < level.price && current.close > level.price && current.close > current.open);
    if (candidate) {
      pending = {
        direction:candidate.type==="high"?"SELL":"BUY",
        levelName:candidate.name,
        level:candidate.price,
        sneakyIndex:index,
        sneakyHigh:current.high,
        sneakyLow:current.low,
        confirmed:false,
        confirmationIndex:null
      };
    }
  }
  return {setups, daily, reason:"Conservative three-candle adaptation; informational only."};
}
function updateSetupHistory() {
  const now = feedNowSeconds();
  if (!Number.isFinite(now)) return;
  const knownIds = new Set(setupHistory.map(setup => setup.id));
  for (const signal of signalsForSeries()) {
    if (!knownIds.has(signal.id)) {
      setupHistory.push(signal);
      knownIds.add(signal.id);
    }
  }
  for (const setup of setupHistory) {
    setup.index = candles.findIndex(item => item.time === setup.candleTime);
    if (setup.outcome !== "No level reached yet") continue;
    for (const candle of candles) {
      if (candle.time < setup.signalTime || candle.time + config.granularity > now) continue;
      const stopHit = setup.direction === "BUY"
        ? candle.low <= setup.stop : candle.high >= setup.stop;
      const targetHit = setup.direction === "BUY"
        ? candle.high >= setup.target : candle.low <= setup.target;
      if (stopHit && targetHit) {
        setup.outcome = "Both levels touched; order unknown";
        break;
      }
      if (targetHit) {
        setup.outcome = "Target reached (chart)";
        break;
      }
      if (stopHit) {
        setup.outcome = "Stop reached (chart)";
        break;
      }
    }
  }
  if (setupHistory.length > 100) setupHistory.splice(0, setupHistory.length - 100);
}
function updateRulerControls() {
  const max = Math.max(0, candles.length - 1);
  startSlider.max = String(max);
  endSlider.max = String(max);
  if (!initializedRuler && candles.length) {
    startSlider.value = String(max);
    endSlider.value = String(max);
    initializedRuler = true;
  } else {
    startSlider.value = String(clamp(Number(startSlider.value), 0, max));
    endSlider.value = String(clamp(Number(endSlider.value), 0, max));
  }
  updateRulerLabels();
}
function updateRulerLabels() {
  const a = candles[Number(startSlider.value)];
  const b = candles[Number(endSlider.value)];
  document.getElementById("startValue").textContent = a ? fmt(a.close) : "—";
  document.getElementById("endValue").textContent = b ? fmt(b.close) : "—";
  document.getElementById("change").textContent = a && b
    && Number(startSlider.value) !== Number(endSlider.value)
    ? `${fmt(b.close - a.close)} (${Math.abs(Number(endSlider.value) - Number(startSlider.value))} bars)`
    : "—";
}
function drawText(text, x, y, color="#aeb4c2", align="left") {
  ctx.fillStyle = color;
  ctx.textAlign = align;
  ctx.fillText(text, x, y);
}
function drawChart() {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(rect.width*dpr) || canvas.height !== Math.round(rect.height*dpr)) {
    canvas.width = Math.round(rect.width*dpr);
    canvas.height = Math.round(rect.height*dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const profileWidth=config.analysisMode!=="Existing analysis"&&config.volumeProfile?120:0;
  const width=rect.width, height=rect.height, left=76, right=18+profileWidth, top=18, bottom=44;
  ctx.clearRect(0,0,width,height);
  ctx.fillStyle="#0e1117"; ctx.fillRect(0,0,width,height);
  if (!candles.length) { drawText("Waiting for live candles…", left, 36); return; }
  const end=clamp(viewEnd,0,candles.length-1);
  const start=Math.max(0,end-visibleCount+1);
  const visible=candles.slice(start,end+1);
  if (!visible.length) return;
  const signals=setupHistory;
  const structure=marketStructureForSeries();
  const supplyDemand=supplyDemandForSeries();
  const sneaky=sneakyPivotForSeries(structure);
  const latest=signals.length ? signals[signals.length-1] : null;
  const latestSneaky=sneaky.setups.length ? sneaky.setups[sneaky.setups.length-1] : null;
  const latestSetupVisible=latest && latest.index>=start && latest.index<=end ? latest : null;
  const visibleSwings=structure.swings.filter(swing=>swing.index>=start&&swing.index<=end).slice(-2);
  const visibleBreaks=structure.breaks.filter(event=>event.index>=start&&event.index<=end).slice(-1);
  const visibleSweeps=structure.sweeps.filter(event=>event.index>=start&&event.index<=end).slice(-1);
  const rulerA=candles[Number(startSlider.value)];
  const rulerB=candles[Number(endSlider.value)];
  let minPrice=Math.min(...visible.map(item=>item.low));
  let maxPrice=Math.max(...visible.map(item=>item.high));
  const visiblePriceRange=maxPrice-minPrice;
  const padding=visiblePriceRange*0.08 || Math.abs(maxPrice)*0.001 || 1;
  minPrice-=padding; maxPrice+=padding;
  const plotW=width-left-right, plotH=height-top-bottom, slot=plotW/visibleCount;
  const xFor=i=>left+(i-start+0.5)*slot;
  const yFor=price=>top+(maxPrice-price)/(maxPrice-minPrice)*plotH;
  ctx.font="12px Arial";
  for (const zone of [supplyDemand.latestDemand,supplyDemand.latestSupply]) {
    if (!zone || zone.high<minPrice || zone.low>maxPrice) continue;
    const zoneStart=Math.max(start,zone.originIndex);
    const zoneEnd=Math.max(zoneStart,end);
    const topY=Math.max(top,yFor(zone.high));
    const bottomY=Math.min(top+plotH,yFor(zone.low));
    if(bottomY<=topY)continue;
    const demand=zone.direction==="Demand";
    const opacity=zone.strengthScore>=80?".18":zone.strengthScore>=65?".12":".07";
    ctx.fillStyle=demand
      ?`rgba(38,198,218,${opacity})`:`rgba(255,138,101,${opacity})`;
    ctx.fillRect(
      xFor(zoneStart)-slot/2,topY,
      Math.max(slot,(zoneEnd-zoneStart+1)*slot),bottomY-topY
    );
    drawText(
      `${demand?"DEMAND":"SUPPLY"} · ${zone.strength.toUpperCase()} · ${zone.touches} TEST${zone.touches===1?"":"S"}`,
      width-right-4,(topY+bottomY)/2,
      demand?"#26c6da":"#ff8a65","right"
    );
  }
  for(let line=0;line<=5;line++){
    const y=top+plotH*line/5;
    const val=maxPrice-(maxPrice-minPrice)*line/5;
    ctx.strokeStyle="rgba(128,132,149,.24)";ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(width-right,y);ctx.stroke();
    drawText(fmt(val),left-8,y+4,"#c4cad5","right");
  }
  if(profileWidth&&config.volumeProfile.bins.length){
    const maxTicks=Math.max(...config.volumeProfile.bins.map(bin=>bin.tick_count));
    for(const bin of config.volumeProfile.bins){
      const upper=yFor(bin.price+config.volumeProfile.binSize/2);
      const lower=yFor(bin.price-config.volumeProfile.binSize/2);
      if(lower<top||upper>top+plotH)continue;
      const barWidth=(profileWidth-12)*bin.tick_count/maxTicks;
      ctx.fillStyle=bin.tick_count===maxTicks?"rgba(255,202,40,.78)":"rgba(66,165,245,.58)";
      ctx.fillRect(width-right+4,upper,barWidth,Math.max(1,lower-upper));
    }
    drawText("TICK ACTIVITY",width-18,top+12,"#aeb4c2","right");
  }
  const candleWidth=Math.max(2,Math.min(12,slot*0.72));
  visible.forEach((item,offset)=>{
    const index=start+offset,x=xFor(index),openY=yFor(item.open),closeY=yFor(item.close);
    const up=item.close>=item.open,color=up?"#26a69a":"#ef5350";
    ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(x,yFor(item.high));ctx.lineTo(x,yFor(item.low));ctx.stroke();
    const bodyY=Math.min(openY,closeY),bodyH=Math.max(1,Math.abs(closeY-openY));
    ctx.fillRect(x-candleWidth/2,bodyY,candleWidth,bodyH);
  });
  visibleSwings.forEach(swing=>{
    const isHigh=swing.label==="HH"||swing.label==="LH"||swing.label==="SH";
    ctx.fillStyle="#b39ddb";
    ctx.beginPath();
    ctx.arc(xFor(swing.index),yFor(swing.price),2.5,0,Math.PI*2);
    ctx.fill();
    drawText(swing.label,xFor(swing.index),yFor(swing.price)+(isHigh?-8:16),"#b39ddb","center");
  });
  visibleBreaks.forEach((event,index)=>{
    const item=candles[event.index];
    const y=yFor(item.close)-10;
    ctx.fillStyle=event.label.startsWith("CHoCH")?"#ffca28":"#80cbc4";
    ctx.beginPath();
    ctx.arc(xFor(event.index),y,3,0,Math.PI*2);
    ctx.fill();
    if(index===visibleBreaks.length-1){
      drawText(event.label,xFor(event.index),y-7,
        event.label.startsWith("CHoCH")?"#ffca28":"#80cbc4","center");
    }
  });
  visibleSweeps.forEach((event,index)=>{
    const color=event.direction==="Bullish"?"#26c6da":"#ff8a65";
    ctx.fillStyle=color;
    ctx.beginPath();
    ctx.arc(xFor(event.index),yFor(event.price),3,0,Math.PI*2);
    ctx.fill();
    if(index===visibleSweeps.length-1){
      drawText(event.label,xFor(event.index),
        yFor(event.price)+(event.direction==="Bullish"?18:-10),color,"center");
    }
  });
  for(let label=0;label<=5;label++){
    const index=Math.round(start+(end-start)*label/5);
    const item=candles[index]; if(!item) continue;
    const date=new Date(item.time*1000);
    const text=date.toLocaleTimeString([], {hour:"2-digit",minute:"2-digit"});
    ctx.strokeStyle="rgba(128,132,149,.18)";ctx.beginPath();
    ctx.moveTo(xFor(index),top);ctx.lineTo(xFor(index),top+plotH);ctx.stroke();
    drawText(text,xFor(index),height-17,"#c4cad5","center");
  }
  function level(name,value,color){
    if(value<minPrice||value>maxPrice)return;
    const y=yFor(value);ctx.strokeStyle=color;ctx.setLineDash([6,5]);
    ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(width-right,y);ctx.stroke();ctx.setLineDash([]);
    drawText(`${name}: ${fmt(value)}`,left+5,y-5,color);
  }
  if(latestSetupVisible){
    level("Entry",latestSetupVisible.entry,"#29b6f6");
    level("Stop",latestSetupVisible.stop,"#ef5350");
    level("Target",latestSetupVisible.target,"#66bb6a");
  }
  if(config.analysisMode!=="Existing analysis"&&config.volumeProfile){
    level("POC",config.volumeProfile.poc,"#ffca28");
    level("VAH",config.volumeProfile.vah,"#42a5f5");
    level("VAL",config.volumeProfile.val,"#42a5f5");
  }
  signals.filter(signal=>signal.index>=start&&signal.index<=end).forEach(signal=>{
    const x=xFor(signal.index),y=yFor(signal.entry),up=signal.direction==="BUY";
    ctx.fillStyle=up?"#00c853":"#ef5350";
    ctx.beginPath();
    ctx.moveTo(x,y+(up?8:-8));ctx.lineTo(x-7,y+(up?-5:5));ctx.lineTo(x+7,y+(up?-5:5));ctx.closePath();ctx.fill();
    if(signal.id===latest?.id){
      drawText(signal.direction,x,y+(up?-10:17),up?"#00e676":"#ff6e6e","center");
    }
  });
  if(latestSneaky&&latestSneaky.index>=start&&latestSneaky.index<=end){
    const x=xFor(latestSneaky.index),y=yFor(latestSneaky.entry);
    ctx.fillStyle="#fbc02d";
    ctx.beginPath();
    ctx.arc(x,y,4,0,Math.PI*2);
    ctx.fill();
    drawText(`SP ${latestSneaky.direction}`,x,y+(latestSneaky.direction==="BUY"?-10:18),
      "#fbc02d","center");
  }
  if(rulerA&&rulerB&&Number(startSlider.value)!==Number(endSlider.value)){
    const ia=Number(startSlider.value),ib=Number(endSlider.value);
    ctx.strokeStyle="#fbc02d";ctx.lineWidth=2;ctx.setLineDash([4,4]);
    ctx.beginPath();ctx.moveTo(xFor(ia),yFor(rulerA.close));ctx.lineTo(xFor(ib),yFor(rulerB.close));ctx.stroke();ctx.setLineDash([]);
    for(const [i,item] of [[ia,rulerA],[ib,rulerB]]){ctx.fillStyle="#fbc02d";ctx.beginPath();ctx.arc(xFor(i),yFor(item.close),4,0,Math.PI*2);ctx.fill();}
  }
  document.getElementById("direction").textContent=latest?latest.direction:"No signal";
  document.getElementById("entry").textContent=latest?fmt(latest.entry):"—";
  document.getElementById("stop").textContent=latest?fmt(latest.stop):"—";
  document.getElementById("target").textContent=latest?fmt(latest.target):"—";
  document.getElementById("signal").textContent=serverTimeOffsetSeconds===null
    ? "Synchronizing with market server time before analyzing candle closes."
    : latest
    ? `${latest.direction} breakout · Confirmed at candle close ${new Date(latest.signalTime*1000).toLocaleString()} · ${latest.outcome} · Informational only; no broker position is tracked.`
    : "Scanning live candles. A signal is confirmed only after its candle closes; no trades are placed.";
  document.getElementById("structure").textContent=serverTimeOffsetSeconds===null
    ? "Market structure: synchronizing with market server time."
    : `Market structure (${structure.bias}) · confirmed swings: ${structure.swings.length} · `+
      `${structure.breaks.length ? "latest " + structure.breaks[structure.breaks.length-1].label : "no confirmed break"}. `+
      "Swing points use two closed candles on each side.";
  const latestSweep=structure.sweeps[structure.sweeps.length-1];
  document.getElementById("liquidity").textContent=latestSweep
    ? `Liquidity sweeps: ${structure.sweeps.length} confirmed · latest ${latestSweep.label} at ${fmt(latestSweep.level)}. `+
      "A closed-candle wick must cross the latest confirmed swing and close back across it; each level is marked once."
    : "Liquidity sweeps: none confirmed. A closed-candle wick must cross the latest confirmed swing and close back across it.";
  const zoneSummary=zone=>zone
    ? `${zone.direction} ${zone.strength.toLowerCase()} (${zone.strengthScore}/100), `+
      `${zone.status.toLowerCase()}, ${zone.touches} test${zone.touches===1?"":"s"} `+
      `(${fmt(zone.low)}–${fmt(zone.high)})`
    : null;
  const zoneParts=[zoneSummary(supplyDemand.latestDemand),zoneSummary(supplyDemand.latestSupply)]
    .filter(Boolean);
  document.getElementById("zones").textContent=zoneParts.length
    ? `Supply/demand: ${zoneParts.join(" · ")}. Only the latest active zones are shaded.`
    : "Supply/demand: no active zones. Requires three strong candles and sufficient displacement.";
  document.getElementById("sneaky").textContent=latestSneaky
    ? `Sneaky Pivot (15m adaptation): ${latestSneaky.direction} · ${latestSneaky.levelName} swept · entry ${fmt(latestSneaky.entry)}, stop ${fmt(latestSneaky.stop)}, target ${fmt(latestSneaky.target)} · confirmed ${new Date(latestSneaky.signalTime*1000).toLocaleString()}.`
    : `Sneaky Pivot (15m adaptation): ${sneaky.reason}`+
      (sneaky.daily
        ? ` Previous-day high ${fmt(sneaky.daily.high)} · low ${fmt(sneaky.daily.low)}.`
        : "");
  document.getElementById("history").innerHTML=signals.length
    ? `<table><thead><tr><th>Confirmed time</th><th>Signal</th><th>Entry</th><th>Stop</th><th>Target</th><th>Chart status</th></tr></thead><tbody>${signals.slice().reverse().map(setup=>`<tr><td>${new Date(setup.signalTime*1000).toLocaleString()}</td><td>${setup.direction} · Confirmed</td><td>${fmt(setup.entry)}</td><td>${fmt(setup.stop)}</td><td>${fmt(setup.target)}</td><td>${setup.outcome}</td></tr>`).join("")}</tbody></table>`
    : "No confirmed setups in the loaded candles.";
  updateRulerLabels();
}
function render(){
  if(renderScheduled)return;
  renderScheduled=true;
  requestAnimationFrame(()=>{
    renderScheduled=false;
    drawChart();
  });
}
function connect() {
  clearTimeout(retryTimer);
  clearTimeout(timeSyncTimer);
  serverTimeOffsetSeconds = null;
  historyRequestId = null;
  dailySubscriptionId = null;
  ws=new WebSocket("wss://api.derivws.com/trading/v1/options/ws/public");
  ws.onopen=()=>{
    const intervalLabel=config.granularity===3600
      ?"1-hour candles":`${config.granularity/60}-minute candles`;
    dailySubscriptionId=null;
    setStatus(`Connected · ${config.asset} · ${intervalLabel}`);
    requestServerTime();
    if(config.granularity===900){
      ws.send(JSON.stringify({
        ticks_history:config.symbol,end:"latest",granularity:86400,
        count:2,style:"candles",subscribe:1,req_id:2
      }));
    }
  };
  ws.onmessage=event=>{
    let message;
    try { message=JSON.parse(event.data); } catch { setStatus("Received unreadable market data.",true); return; }
    if (Number(message.req_id) === 3 && Number.isFinite(Number(message.time))) {
      serverTimeOffsetSeconds = Number(message.time) - Date.now() / 1000;
      feedClockError = "";
      mergeRows([]);
      if (historyRequestId === null) {
        historyWindowStart = tradingWindowStart(feedNowSeconds());
        previousHistoryOldest = null;
        requestHistoryPage("latest");
      }
      return;
    }
    if (historyRequestId !== null
        && Number(message.req_id) === historyRequestId) {
      if (message.error) {
        historyRequestId = null;
        setStatus(`Historical candle request failed: ${message.error.message}`, true);
        finishHistoryLoad();
        return;
      }
      const rows = message.candles || [];
      if (!rows.length) {
        finishHistoryLoad();
        return;
      }
      mergeRows(rows);
      const oldest = Math.min(...rows.map(row =>
        Number(row.open_time ?? row.epoch)
      ));
      if (oldest > historyWindowStart && oldest !== previousHistoryOldest) {
        previousHistoryOldest = oldest;
        requestHistoryPage(oldest - 1);
      } else {
        if (oldest > historyWindowStart) {
          setStatus(
            "Market feed returned no earlier candles; displaying available history.",
            true
          );
        }
        finishHistoryLoad();
      }
      return;
    }
    if(message.error){
      if(Number(message.req_id)===2){
        dailyFeedError=message.error.message;
        render();
      }else if(Number(message.req_id)===3){
        feedClockError=`Market server time unavailable: ${message.error.message}`;
        setStatus(feedClockError,true);
      }else setStatus(`Deriv feed error: ${message.error.message}`,true);
      return;
    }
    if(Number(message.req_id)===2
        || (dailySubscriptionId && message.subscription?.id===dailySubscriptionId)){
      if(message.subscription?.id)dailySubscriptionId=message.subscription.id;
      if(message.candles?.length)mergePreviousDayRows(message.candles);
      else if(message.ohlc)mergePreviousDayRows([message.ohlc]);
      return;
    }
    if(message.candles?.length){
      mergeRows(message.candles);
      setStatus(
        feedClockError || `Live · ${config.asset} · last update ${new Date().toLocaleTimeString()}`,
        Boolean(feedClockError)
      );
    } else if(message.ohlc){
      mergeRows([message.ohlc]);
      setStatus(
        feedClockError || `Live · ${config.asset} · last update ${new Date().toLocaleTimeString()}`,
        Boolean(feedClockError)
      );
    }
  };
  ws.onerror=()=>setStatus("Live feed connection error; retrying…",true);
  ws.onclose=()=>{
    setStatus("Live feed disconnected; reconnecting in 5 seconds…",true);
    retryTimer=setTimeout(connect,5000);
  };
}
function requestHistoryPage(end) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  historyRequestId = nextHistoryRequestId++;
  setStatus("Loading recent chart candle history…");
  ws.send(JSON.stringify({
    ticks_history:config.symbol,
    start:historyWindowStart,
    end,
    granularity:config.granularity,
    count:HISTORY_PAGE_SIZE,
    style:"candles",
    req_id:historyRequestId
  }));
}
function finishHistoryLoad() {
  historyRequestId = null;
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({
    ticks_history:config.symbol,
    end:"latest",
    granularity:config.granularity,
    count:1,
    style:"candles",
    subscribe:1,
    req_id:4
  }));
}
function requestServerTime() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({time: 1, req_id: 3}));
  timeSyncTimer = setTimeout(requestServerTime, 60000);
}
if (Array.isArray(config.initialCandles) && config.initialCandles.length) {
  mergeRows(config.initialCandles);
}
document.getElementById("pan").onclick=()=>{mode="pan";document.getElementById("pan").classList.add("active");document.getElementById("zoom").classList.remove("active");};
document.getElementById("zoom").onclick=()=>{mode="zoom";document.getElementById("zoom").classList.add("active");document.getElementById("pan").classList.remove("active");};
document.getElementById("zoomIn").onclick=()=>{visibleCount=Math.max(8,Math.floor(visibleCount*0.8));viewEnd=candles.length-1;render();};
document.getElementById("zoomOut").onclick=()=>{visibleCount=Math.min(Math.max(60,candles.length),Math.ceil(visibleCount*1.25));viewEnd=candles.length-1;render();};
document.getElementById("reset").onclick=()=>{mode="pan";visibleCount=60;viewEnd=candles.length-1;document.getElementById("pan").classList.add("active");document.getElementById("zoom").classList.remove("active");render();};
canvas.addEventListener("pointerdown",event=>{dragStart={x:event.clientX,viewEnd,mode};canvas.setPointerCapture(event.pointerId);});
canvas.addEventListener("pointermove",event=>{
  if(!dragStart)return;
  const rect=canvas.getBoundingClientRect(),step=rect.width/visibleCount;
  if(dragStart.mode==="pan")viewEnd=clamp(dragStart.viewEnd-Math.round((event.clientX-dragStart.x)/step),Math.min(visibleCount-1,candles.length-1),candles.length-1);
  else {const delta=Math.abs(event.clientX-dragStart.x);if(delta>20)visibleCount=clamp(Math.round(visibleCount*rect.width/delta),8,Math.max(60,candles.length));}
  render();
});
canvas.addEventListener("pointerup",()=>{dragStart=null;});
canvas.addEventListener("wheel",event=>{
  event.preventDefault();visibleCount=clamp(Math.round(visibleCount*(event.deltaY<0?0.9:1.1)),8,Math.max(60,candles.length));
  viewEnd=clamp(viewEnd,Math.min(visibleCount-1,candles.length-1),candles.length-1);render();
},{passive:false});
startSlider.addEventListener("input",render);
endSlider.addEventListener("input",render);
new ResizeObserver(render).observe(canvas);
connect();
</script>
</body>
</html>
""".replace("__LIVE_CONFIG__", config)


# -----------------------------------------------------------------------------
# 4. SIDEBAR CONTROLS REGISTRATION (FIXED INTERFACE DECLARATIONS)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("System Mode Switch")
    data_mode = st.radio("Data Sourcing Mode", ["Local Pattern Simulator", "Live WebSocket Feed"])
    if data_mode == "Live WebSocket Feed":
        st.caption("Live mode uses read-only candles. It does not place or manage trades.")

    st.markdown("---")
    st.title("Market Data Connection")
    st.caption(
        "Live candles use Deriv's public market-data connection. No App ID, account "
        "password, or API token is needed; this connection reads prices only and "
        "cannot place trades."
    )
    
    st.markdown("---")
    st.title("Market Matrix")
    asset_selection = st.selectbox("Active Asset", ["Gold (XAUUSD)", "EURUSD", "GBPUSD"])
    timeframe_selection = st.selectbox(
        "Interval Window",
        list(TIMEFRAME_SECONDS),
        index=1,
    )
    analysis_mode = st.radio(
        "Analysis Mode",
        ["Existing analysis", "Volume Profile", "Combined"],
        horizontal=True,
        help=(
            "Combined mode keeps existing breakout confirmation and additionally "
            "requires a close beyond the previous session's value area."
        ),
    )
    
    st.markdown("---")
    st.title("Account Balance Mod")
    starting_balance = st.number_input("Current Balance ($)", min_value=0.0, value=100.0, step=10.0)
    risk_percentage = st.slider("Risk Per Trade (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.5)

    st.markdown("---")
    st.title("Target Logic Extension")
    aggressive_target = st.checkbox("Enable Aggressive Target", value=False, help="Switches to wide structural extension multipliers.")
    round_trip_cost_r = st.slider(
        "Backtest estimated round-trip cost (R)",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help=(
            "Deduct this user-entered R cost from each completed backtest trade. "
            "Zero means costs are not included; this is not a broker quote."
        ),
    )

    st.markdown("---")
    st.info("👤 **Active Account Identity:**\n\n**B. MARATI (Bliss)**\n\n*Status: System Live*")

# -----------------------------------------------------------------------------
# 5. AUTOMATIC CANDLE UPDATES AND SIGNAL DETECTION
# -----------------------------------------------------------------------------
if "order_logs" not in st.session_state:
    st.session_state["order_logs"] = []
if "engine_status" not in st.session_state:
    st.session_state["engine_status"] = "Initializing"


def calculate_auto_signals(
    df,
    lookback=20,
    target_multiple=2.0,
    granularity_seconds=300,
):
    """Return confirmed breakout markers and their independent chart outcomes."""
    df = df.copy().sort_values("Time").reset_index(drop=True)
    candle_count = len(df)
    triggers = np.full(candle_count, "HOLD", dtype=object)
    entries = np.full(candle_count, np.nan)
    stop_losses = np.full(candle_count, np.nan)
    profit_targets = np.full(candle_count, np.nan)
    signal_statuses = np.full(candle_count, "", dtype=object)
    highs = df["High"].to_numpy(dtype=float, copy=False)
    lows = df["Low"].to_numpy(dtype=float, copy=False)
    opens = df["Open"].to_numpy(dtype=float, copy=False)
    closes = df["Close"].to_numpy(dtype=float, copy=False)
    times = df["Time"].to_numpy(copy=False)

    if candle_count < lookback + 2:
        df["Trigger"] = triggers
        df["Entry"] = entries
        df["Stop Loss"] = stop_losses
        df["Profit Target"] = profit_targets
        df["Signal Status"] = signal_statuses
        return df, None, []

    latest_signal = None
    setups = []
    last_signal_by_direction = {}
    for index in range(lookback + 1, candle_count):
        reference_start = index - lookback - 1
        reference_end = index - 1
        previous_start = index - lookback
        entry = closes[index]
        previous_close = closes[index - 1]
        resistance = float(np.max(highs[reference_start:reference_end]))
        support = float(np.min(lows[reference_start:reference_end]))

        if (
            previous_close <= resistance
            and entry > resistance
            and entry > opens[index]
        ):
            direction = "BUY"
            stop_loss = float(np.min(lows[max(previous_start, index - 10):index]))
            risk_distance = entry - stop_loss
            target_direction = 1
        elif (
            previous_close >= support
            and entry < support
            and entry < opens[index]
        ):
            direction = "SELL"
            stop_loss = float(np.max(highs[max(previous_start, index - 10):index]))
            risk_distance = stop_loss - entry
            target_direction = -1
        else:
            continue

        if risk_distance <= 0:
            continue

        if index - last_signal_by_direction.get(direction, -lookback) < lookback:
            continue

        profit_target = entry + target_direction * risk_distance * target_multiple
        triggers[index] = direction
        entries[index] = entry
        stop_losses[index] = stop_loss
        profit_targets[index] = profit_target
        last_signal_by_direction[direction] = index
        signal_time = pd.Timestamp(times[index]) + pd.Timedelta(
            seconds=granularity_seconds
        )
        setup = {
            "direction": direction,
            "entry": entry,
            "stop_loss": stop_loss,
            "profit_target": profit_target,
            "risk_distance": risk_distance,
            "signal_time": signal_time,
            "signal_index": index,
            "status": "No level reached yet",
            "confirmation_status": "Confirmed at candle close",
        }
        future_highs = highs[index + 1:]
        future_lows = lows[index + 1:]
        if direction == "BUY":
            stop_hits = future_lows <= stop_loss
            target_hits = future_highs >= profit_target
        else:
            stop_hits = future_highs >= stop_loss
            target_hits = future_lows <= profit_target
        first_hit = np.flatnonzero(stop_hits | target_hits)
        if first_hit.size:
            outcome_offset = first_hit[0]
            stop_hit = stop_hits[outcome_offset]
            target_hit = target_hits[outcome_offset]
            if stop_hit and target_hit:
                setup["status"] = "Both levels touched; order unknown"
            elif target_hit:
                setup["status"] = "Target reached (chart)"
            else:
                setup["status"] = "Stop reached (chart)"
        signal_statuses[index] = setup["confirmation_status"]
        setups.append(setup)
        latest_signal = {
            **setup,
        }

    df["Trigger"] = triggers
    df["Entry"] = entries
    df["Stop Loss"] = stop_losses
    df["Profit Target"] = profit_targets
    df["Signal Status"] = signal_statuses
    return df, latest_signal, setups


def calculate_market_structure(df, pivot_bars=2, granularity_seconds=300):
    """Label confirmed swing pivots and close-confirmed breaks of structure."""
    df = df.copy().sort_values("Time").reset_index(drop=True)
    df["Swing High"] = np.nan
    df["Swing Low"] = np.nan
    df["Swing Label"] = ""
    df["Structure Event"] = ""
    df["Liquidity Sweep"] = ""

    events = []
    swing_highs = []
    swing_lows = []
    latest_swing_high = None
    latest_swing_low = None
    broken_high_index = None
    broken_low_index = None
    swept_high_indices = set()
    swept_low_indices = set()
    bias = "Neutral"

    for index in range(1, len(df)):
        current = df.iloc[index]
        previous = df.iloc[index - 1]

        if (
            latest_swing_high is not None
            and broken_high_index != latest_swing_high["index"]
            and float(previous["Close"]) <= latest_swing_high["price"]
            and float(current["Close"]) > latest_swing_high["price"]
        ):
            event_label = "CHoCH UP" if bias == "Bearish" else "BOS UP"
            df.loc[index, "Structure Event"] = event_label
            events.append({
                "type": "break",
                "label": event_label,
                "price": latest_swing_high["price"],
                "index": index,
                "time": current["Time"],
                "confirmed_at": current["Time"] + pd.Timedelta(
                    seconds=granularity_seconds
                ),
            })
            bias = "Bullish"
            broken_high_index = latest_swing_high["index"]

        if (
            latest_swing_low is not None
            and broken_low_index != latest_swing_low["index"]
            and float(previous["Close"]) >= latest_swing_low["price"]
            and float(current["Close"]) < latest_swing_low["price"]
        ):
            event_label = "CHoCH DOWN" if bias == "Bullish" else "BOS DOWN"
            df.loc[index, "Structure Event"] = event_label
            events.append({
                "type": "break",
                "label": event_label,
                "price": latest_swing_low["price"],
                "index": index,
                "time": current["Time"],
                "confirmed_at": current["Time"] + pd.Timedelta(
                    seconds=granularity_seconds
                ),
            })
            bias = "Bearish"
            broken_low_index = latest_swing_low["index"]

        if (
            latest_swing_low is not None
            and latest_swing_low["index"] not in swept_low_indices
            and float(current["Low"]) < latest_swing_low["price"]
            and float(current["Close"]) > latest_swing_low["price"]
        ):
            event = {
                "type": "sweep",
                "direction": "Bullish",
                "label": "SSL sweep",
                "price": float(current["Low"]),
                "level": latest_swing_low["price"],
                "index": index,
                "time": current["Time"],
                "confirmed_at": current["Time"] + pd.Timedelta(
                    seconds=granularity_seconds
                ),
            }
            df.loc[index, "Liquidity Sweep"] = event["label"]
            events.append(event)
            swept_low_indices.add(latest_swing_low["index"])

        if (
            latest_swing_high is not None
            and latest_swing_high["index"] not in swept_high_indices
            and float(current["High"]) > latest_swing_high["price"]
            and float(current["Close"]) < latest_swing_high["price"]
        ):
            event = {
                "type": "sweep",
                "direction": "Bearish",
                "label": "BSL sweep",
                "price": float(current["High"]),
                "level": latest_swing_high["price"],
                "index": index,
                "time": current["Time"],
                "confirmed_at": current["Time"] + pd.Timedelta(
                    seconds=granularity_seconds
                ),
            }
            df.loc[index, "Liquidity Sweep"] = (
                f"{df.loc[index, 'Liquidity Sweep']}/{event['label']}"
                if df.loc[index, "Liquidity Sweep"]
                else event["label"]
            )
            events.append(event)
            swept_high_indices.add(latest_swing_high["index"])

        pivot_index = index - pivot_bars
        if pivot_index < pivot_bars:
            continue
        left = pivot_index - pivot_bars
        window = df.iloc[left:index + 1]
        pivot = df.iloc[pivot_index]
        highest = float(window["High"].max())
        lowest = float(window["Low"].min())
        pivot_time = pivot["Time"]
        confirmed_at = current["Time"] + pd.Timedelta(
            seconds=granularity_seconds
        )
        new_swing = False

        if (
            float(pivot["High"]) == highest
            and int((window["High"] == highest).sum()) == 1
        ):
            previous_high = swing_highs[-1] if swing_highs else None
            label = (
                "SH" if previous_high is None
                else "HH" if highest > previous_high["price"]
                else "LH"
            )
            swing = {
                "type": "swing_high",
                "label": label,
                "price": highest,
                "index": pivot_index,
                "time": pivot_time,
                "confirmed_at": confirmed_at,
            }
            swing_highs.append(swing)
            latest_swing_high = swing
            new_swing = True
            df.loc[pivot_index, "Swing High"] = highest
            df.loc[pivot_index, "Swing Label"] = label
            events.append(swing)

        if (
            float(pivot["Low"]) == lowest
            and int((window["Low"] == lowest).sum()) == 1
        ):
            previous_low = swing_lows[-1] if swing_lows else None
            label = (
                "SL" if previous_low is None
                else "HL" if lowest > previous_low["price"]
                else "LL"
            )
            swing = {
                "type": "swing_low",
                "label": label,
                "price": lowest,
                "index": pivot_index,
                "time": pivot_time,
                "confirmed_at": confirmed_at,
            }
            swing_lows.append(swing)
            latest_swing_low = swing
            new_swing = True
            df.loc[pivot_index, "Swing Low"] = lowest
            existing_label = str(df.loc[pivot_index, "Swing Label"])
            df.loc[pivot_index, "Swing Label"] = (
                f"{existing_label}/{label}" if existing_label else label
            )
            events.append(swing)

        if new_swing:
            latest_high_label = swing_highs[-1]["label"] if swing_highs else ""
            latest_low_label = swing_lows[-1]["label"] if swing_lows else ""
            if latest_high_label == "HH" and latest_low_label == "HL":
                bias = "Bullish"
            elif latest_high_label == "LH" and latest_low_label == "LL":
                bias = "Bearish"

    return df, events, bias


def calculate_sneaky_pivot_setups(
    df,
    previous_day_high,
    previous_day_low,
    structure_events,
    target_multiple=1.0,
    granularity_seconds=900,
):
    """Find conservative 15-minute Sneaky Pivot setups from confirmed candles."""
    if (
        granularity_seconds != 900
        or previous_day_high is None
        or previous_day_low is None
    ):
        return []

    df = df.copy().sort_values("Time").reset_index(drop=True)
    swings = [
        event for event in structure_events
        if event["type"] in ("swing_high", "swing_low")
    ]
    setups = []
    pending = None

    for index in range(1, len(df)):
        current = df.iloc[index]
        previous = df.iloc[index - 1]
        if pending is not None:
            if not pending["confirmed"]:
                confirmation = (
                    float(current["Close"]) < float(current["Open"])
                    and float(current["Close"]) < pending["level"]
                    if pending["direction"] == "SELL"
                    else float(current["Close"]) > float(current["Open"])
                    and float(current["Close"]) > pending["level"]
                )
                if confirmation:
                    pending["confirmed"] = True
                    pending["confirmation_index"] = index
                else:
                    pending = None
                continue

            if index <= pending["confirmation_index"]:
                continue
            invalidated = (
                float(current["Close"]) > pending["sneaky_high"]
                if pending["direction"] == "SELL"
                else float(current["Close"]) < pending["sneaky_low"]
            )
            if invalidated:
                pending = None
                continue
            broken = (
                float(previous["Close"]) >= pending["sneaky_low"]
                and float(current["Close"]) < pending["sneaky_low"]
                if pending["direction"] == "SELL"
                else float(previous["Close"]) <= pending["sneaky_high"]
                and float(current["Close"]) > pending["sneaky_high"]
            )
            if not broken:
                continue

            entry = float(current["Close"])
            stop = pending["sneaky_high"] if pending["direction"] == "SELL" else pending["sneaky_low"]
            risk_distance = abs(entry - stop)
            if risk_distance <= 0:
                pending = None
                continue
            target_direction = -1 if pending["direction"] == "SELL" else 1
            setups.append({
                "direction": pending["direction"],
                "level_name": pending["level_name"],
                "swept_level": pending["level"],
                "sneaky_index": pending["sneaky_index"],
                "signal_index": index,
                "signal_time": pd.Timestamp(current["Time"]) + pd.Timedelta(
                    seconds=granularity_seconds
                ),
                "entry": entry,
                "stop_loss": stop,
                "profit_target": entry + target_direction * risk_distance * target_multiple,
                "confirmation": "Break of sneaky candle after directional confirmation",
            })
            pending = None
            continue

        prior_swings = [
            event for event in swings if int(event["index"]) + 2 < index
        ]
        latest_high = next(
            (
                event for event in reversed(prior_swings)
                if event["type"] == "swing_high"
            ),
            None,
        )
        latest_low = next(
            (
                event for event in reversed(prior_swings)
                if event["type"] == "swing_low"
            ),
            None,
        )
        reference_levels = [
            ("Previous-day high", "swing_high", float(previous_day_high)),
            ("Previous-day low", "swing_low", float(previous_day_low)),
        ]
        if latest_high is not None:
            reference_levels.append(
                (latest_high["label"], "swing_high", float(latest_high["price"]))
            )
        if latest_low is not None:
            reference_levels.append(
                (latest_low["label"], "swing_low", float(latest_low["price"]))
            )
        candidates = []
        for level_name, level_type, level in reference_levels:
            if (
                level_type == "swing_high"
                and float(current["High"]) > level
                and float(current["Close"]) < level
                and float(current["Close"]) < float(current["Open"])
            ):
                candidates.append(("SELL", level_name, level))
            elif (
                level_type == "swing_low"
                and float(current["Low"]) < level
                and float(current["Close"]) > level
                and float(current["Close"]) > float(current["Open"])
            ):
                candidates.append(("BUY", level_name, level))
        if candidates:
            direction, level_name, level = candidates[0]
            pending = {
                "direction": direction,
                "level_name": level_name,
                "level": level,
                "sneaky_index": index,
                "sneaky_high": float(current["High"]),
                "sneaky_low": float(current["Low"]),
                "confirmed": False,
                "confirmation_index": None,
            }

    return setups


# -------------------------------------------------------------------------
# 6. DASHBOARD
# -------------------------------------------------------------------------
def render_dashboard():
    require_private_login()
    st.title("TrigMAR Trading Dashboard")
    if data_mode == "Live WebSocket Feed":
        symbols = {
            "Gold (XAUUSD)": "frxXAUUSD",
            "EURUSD": "frxEURUSD",
            "GBPUSD": "frxGBPUSD",
        }
        symbol = symbols[asset_selection]
        granularity = TIMEFRAME_SECONDS[timeframe_selection]
        ensure_candle_feed(symbol, granularity)
        ensure_tick_feed(symbol)
        initial_candles = load_candle_rows(symbol, granularity)
        chart_cutoff = trading_window_start(
            int(datetime.now(timezone.utc).timestamp()),
            weekdays=CHART_HISTORY_WEEKDAYS,
        )
        chart_initial_candles = [
            row for row in initial_candles if row["epoch"] >= chart_cutoff
        ]
        target_multiple = 2.0 if aggressive_target else 1.0
        backtest_candles = pd.DataFrame([
            {
                "Time": pd.to_datetime(row["epoch"], unit="s", utc=True),
                "Open": row["open"],
                "High": row["high"],
                "Low": row["low"],
                "Close": row["close"],
            }
            for row in initial_candles
        ], columns=["Time", "Open", "High", "Low", "Close"])
        if not backtest_candles.empty:
            backtest_candles = backtest_candles.loc[
                backtest_candles["Time"] + pd.Timedelta(seconds=granularity)
                <= pd.Timestamp.now(tz=timezone.utc)
            ].reset_index(drop=True)
        _, _, baseline_setups = calculate_auto_signals(
            backtest_candles,
            target_multiple=target_multiple,
            granularity_seconds=granularity,
        )
        profile_data = load_previous_session_profile(
            symbol,
            int(datetime.now(timezone.utc).timestamp()),
            asset_selection,
        )
        profile = profile_data["profile"]
        historical_profiles = {}

        def load_backtest_profile(signal_time):
            signal_epoch = int(pd.Timestamp(signal_time).timestamp())
            session_start, _ = previous_utc_session(signal_epoch)
            if session_start not in historical_profiles:
                historical_profiles[session_start] = load_previous_session_profile(
                    symbol,
                    signal_epoch,
                    asset_selection,
                )
            return historical_profiles[session_start]

        combined_setups, unavailable_profile_count = filter_combined_setups(
            backtest_candles,
            baseline_setups,
            load_backtest_profile,
        )
        render_backtest_report(
            backtest_candles,
            baseline_setups,
            target_multiple,
            source_label=(
                f"Deriv live history · {asset_selection} · {timeframe_selection} "
                f"(up to {LIVE_HISTORY_WEEKDAYS} retained weekdays)"
            ),
            granularity_seconds=granularity,
            round_trip_cost_r=round_trip_cost_r,
            combined_setups=combined_setups,
            unavailable_profile_count=unavailable_profile_count,
        )
        st.caption(
            f"{asset_selection} · {timeframe_selection} · "
            "Live read-only Deriv stream. Chart updates in place; no trades are placed."
        )
        st.caption(
            f"Up to {LIVE_HISTORY_WEEKDAYS} weekdays retained · "
            f"{len(initial_candles):,} candles loaded "
            "from local SQLite storage."
        )
        st.caption(f"Local candle database: {CANDLE_DB_PATH}")
        if analysis_mode != "Existing analysis":
            st.subheader("Previous UTC Session Volume Profile")
            st.caption(
                "Tick-activity estimate from observed quote updates, not traded "
                "volume-at-price. Previous completed weekday session; 70% value area."
            )
            if profile is None:
                st.info(f"DATA_UNAVAILABLE · {profile_data['reason']}")
            else:
                profile_decimals = 2 if asset_selection == "Gold (XAUUSD)" else 5
                profile_col1, profile_col2, profile_col3, profile_col4 = st.columns(4)
                profile_col1.metric("POC", f"{profile['poc']:.{profile_decimals}f}")
                profile_col2.metric("VAH", f"{profile['vah']:.{profile_decimals}f}")
                profile_col3.metric("VAL", f"{profile['val']:.{profile_decimals}f}")
                profile_col4.metric("Observed quote ticks", f"{profile_data['tick_count']:,}")
                session_date = datetime.fromtimestamp(
                    profile_data["start"], tz=timezone.utc
                ).strftime("%Y-%m-%d")
                st.caption(
                    f"Session: {session_date} UTC · Estimated activity only. "
                    "Tick counts are not traded volume."
                )
                if profile["nodes"]:
                    st.caption(
                        "Nodes: "
                        + " · ".join(
                            f"{node['type']} {node['price']:.{profile_decimals}f}"
                            for node in profile["nodes"]
                        )
                    )
            if analysis_mode == "Combined":
                st.caption(
                    "Combined filter: the existing closed-candle breakout must also "
                    "close above VAH for BUY or below VAL for SELL. Missing profile "
                    "data means no combined signals."
                )
        st.iframe(
            build_live_chart_html(
                symbol=symbol,
                granularity_seconds=granularity,
                asset=asset_selection,
                analysis_mode=analysis_mode,
                volume_profile=(
                    {
                        "poc": profile["poc"],
                        "vah": profile["vah"],
                        "val": profile["val"],
                        "binSize": profile["bin_size"],
                        "bins": profile["bins"],
                    }
                    if profile is not None
                    else None
                ),
                target_multiple=2.0 if aggressive_target else 1.0,
                starting_balance=starting_balance,
                risk_percentage=risk_percentage,
                initial_candles=chart_initial_candles,
            ),
            height=940,
            alt="TrigMAR live market chart",
        )
        return

    simulator_key = (asset_selection, timeframe_selection)
    if st.session_state.get("simulator_key") != simulator_key:
        interval_minutes = TIMEFRAME_SECONDS[timeframe_selection] // 60
        st.session_state["live_candles"] = generate_simulated_pattern(
            interval_minutes, asset_selection
        )
        st.session_state["simulator_key"] = simulator_key
    st.session_state["engine_status"] = "Simulated candles"
    st.caption(
        f"{asset_selection} · {timeframe_selection} · "
        f"{st.session_state['engine_status']}"
    )

    candles = st.session_state.get("live_candles")
    if candles is None or candles.empty:
        st.info("No simulator candles are available.")
        return

    st.caption("Data source: Local simulator · Prices are generated, not live.")
    target_multiple = 2.0 if aggressive_target else 1.0
    analysis, levels, setups = calculate_auto_signals(
        candles,
        target_multiple=target_multiple,
        granularity_seconds=TIMEFRAME_SECONDS[timeframe_selection],
    )
    render_backtest_report(
        analysis,
        setups,
        target_multiple,
        granularity_seconds=TIMEFRAME_SECONDS[timeframe_selection],
        round_trip_cost_r=round_trip_cost_r,
        source_label=(
            f"Synthetic simulator · {asset_selection} · {timeframe_selection}; "
            "not market-performance evidence"
        ),
    )
    if analysis_mode != "Existing analysis":
        st.info(
            "DATA_UNAVAILABLE · The local simulator generates synthetic OHLC "
            "candles and has no observed quote ticks. Volume Profile and Combined "
            "analysis require the live feed; no tick activity is fabricated."
        )
        return

    analysis, structure_events, structure_bias = calculate_market_structure(
        analysis,
        granularity_seconds=TIMEFRAME_SECONDS[timeframe_selection],
    )
    previous_day_rows = analysis.loc[
        analysis["Time"].dt.date < datetime.now().date()
    ]
    previous_day_high = None
    previous_day_low = None
    if not previous_day_rows.empty:
        previous_day = previous_day_rows["Time"].dt.date.max()
        previous_day_candles = previous_day_rows.loc[
            previous_day_rows["Time"].dt.date == previous_day
        ]
        previous_day_high = float(previous_day_candles["High"].max())
        previous_day_low = float(previous_day_candles["Low"].min())
    sneaky_setups = calculate_sneaky_pivot_setups(
        analysis,
        previous_day_high,
        previous_day_low,
        structure_events,
        target_multiple=target_multiple,
        granularity_seconds=TIMEFRAME_SECONDS[timeframe_selection],
    )
    _, latest_zones = calculate_supply_demand_zones(
        analysis,
        granularity_seconds=TIMEFRAME_SECONDS[timeframe_selection],
    )
    ruler_start, ruler_end = st.slider(
        "Ruler: select start and end candles",
        min_value=0,
        max_value=len(analysis) - 1,
        value=(len(analysis) - 1, len(analysis) - 1),
        key=f"ruler-range-{asset_selection}-{timeframe_selection}",
        help="Select two candle positions to measure the price change and candle count.",
    )
    ruler_start_candle = analysis.iloc[ruler_start]
    ruler_end_candle = analysis.iloc[ruler_end]
    ruler_price_change = float(ruler_end_candle["Close"] - ruler_start_candle["Close"])
    ruler_intervals = ruler_end - ruler_start
    price_decimals = 2 if asset_selection == "Gold (XAUUSD)" else 5

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=analysis["Time"],
        open=analysis["Open"],
        high=analysis["High"],
        low=analysis["Low"],
        close=analysis["Close"],
        name="Price",
        increasing_line_color="#66bb6a",
        increasing_fillcolor="#26a69a",
        decreasing_line_color="#ef5350",
        decreasing_fillcolor="#ef5350",
    ))
    zone_end_time = pd.Timestamp(analysis["Time"].iloc[-1]) + pd.Timedelta(
        seconds=TIMEFRAME_SECONDS[timeframe_selection]
    )
    for direction in ("Demand", "Supply"):
        zone = latest_zones[direction]
        if zone is not None:
            opacity = 0.18 if zone["strength_score"] >= 80 else (
                0.12 if zone["strength_score"] >= 65 else 0.07
            )
            color = (
                f"rgba(38, 198, 218, {opacity})"
                if direction == "Demand"
                else f"rgba(255, 138, 101, {opacity})"
            )
            fig.add_shape(
                type="rect",
                x0=zone["start_time"],
                x1=zone_end_time,
                y0=zone["low"],
                y1=zone["high"],
                fillcolor=color,
                line={"width": 0},
                layer="below",
            )

    signal_rows = analysis[analysis["Trigger"] != "HOLD"]
    latest_signal_index = int(signal_rows.index[-1]) if not signal_rows.empty else None
    for direction, color, symbol in (
        ("BUY", "#00c853", "triangle-up"),
        ("SELL", "#ef5350", "triangle-down"),
    ):
        rows = signal_rows[signal_rows["Trigger"] == direction]
        if not rows.empty:
            fig.add_trace(go.Scatter(
                x=rows["Time"],
                y=rows["Close"],
                mode="markers+text",
                text=[
                    direction if row_index == latest_signal_index else ""
                    for row_index in rows.index
                ],
                textposition="top center" if direction == "BUY" else "bottom center",
                marker={"symbol": symbol, "size": 13, "color": color},
                name=direction,
            ))

    swing_events = [
        event for event in structure_events
        if event["type"] in ("swing_high", "swing_low")
    ][-2:]
    if swing_events:
        fig.add_trace(go.Scatter(
            x=[event["time"] for event in swing_events],
            y=[event["price"] for event in swing_events],
            mode="markers+text",
            text=[event["label"] for event in swing_events],
            textposition=[
                "top center" if event["type"] == "swing_high" else "bottom center"
                for event in swing_events
            ],
            marker={"symbol": "circle", "size": 8, "color": "#b39ddb"},
            name="Confirmed swings (HH/HL/LH/LL)",
            customdata=[event["confirmed_at"] for event in swing_events],
            hovertemplate=(
                "%{text} · %{y:.5f}<br>Pivot: %{x|%Y-%m-%d %H:%M}"
                "<br>Confirmed: %{customdata|%Y-%m-%d %H:%M}<extra></extra>"
            ),
        ))
    all_sweep_events = [
        event for event in structure_events if event["type"] == "sweep"
    ]
    sweep_events = all_sweep_events[-1:]
    if sweep_events:
        fig.add_trace(go.Scatter(
            x=[event["time"] for event in sweep_events],
            y=[event["price"] for event in sweep_events],
            mode="markers+text",
            text=[
                event["label"] if index == len(sweep_events) - 1 else ""
                for index, event in enumerate(sweep_events)
            ],
            textposition=[
                "bottom center" if event["direction"] == "Bullish" else "top center"
                for event in sweep_events
            ],
            marker={
                "symbol": [
                    "triangle-up" if event["direction"] == "Bullish" else "triangle-down"
                    for event in sweep_events
                ],
                "size": 11,
                "color": [
                    "#26c6da" if event["direction"] == "Bullish" else "#ff8a65"
                    for event in sweep_events
                ],
            },
            name="Closed-candle liquidity sweeps",
            customdata=[
                [event["level"], event["confirmed_at"]] for event in sweep_events
            ],
            hovertemplate=(
                "%{text} · wick: %{y:.5f}<br>Swept level: %{customdata[0]:.5f}"
                "<br>Confirmed: %{customdata[1]|%Y-%m-%d %H:%M}<extra></extra>"
            ),
        ))
    all_break_events = [
        event for event in structure_events if event["type"] == "break"
    ]
    break_events = all_break_events[-2:]
    if break_events:
        fig.add_trace(go.Scatter(
            x=[event["time"] for event in break_events],
            y=[event["price"] for event in break_events],
            mode="markers+text",
            text=[
                event["label"] if index == len(break_events) - 1 else ""
                for index, event in enumerate(break_events)
            ],
            textposition="top center",
            marker={"symbol": "diamond", "size": 10, "color": "#ffca28"},
            name="Close-confirmed structure breaks",
            customdata=[event["confirmed_at"] for event in break_events],
            hovertemplate=(
                "%{text} · %{y:.5f}<br>Break candle: %{x|%Y-%m-%d %H:%M}"
                "<br>Confirmed: %{customdata|%Y-%m-%d %H:%M}<extra></extra>"
            ),
        ))

    if ruler_start != ruler_end:
        fig.add_trace(go.Scatter(
            x=[ruler_start_candle["Time"], ruler_end_candle["Time"]],
            y=[ruler_start_candle["Close"], ruler_end_candle["Close"]],
            mode="lines+markers",
            line={"color": "#fbc02d", "width": 2, "dash": "dot"},
            marker={"symbol": "circle", "size": 9, "color": "#fbc02d"},
            name="Ruler",
            hovertemplate=(
                "%{x|%Y-%m-%d %H:%M}<br>Close: %{y:.5f}<extra>Ruler</extra>"
            ),
        ))

    if setups:
        setup = setups[-1]
        signal_time_label = pd.Timestamp(setup["signal_time"]).strftime("%H:%M")
        for name, value, color in (
            (f"Entry {signal_time_label}", setup["entry"], "#29b6f6"),
            (f"Stop {signal_time_label}", setup["stop_loss"], "#ef5350"),
            (f"Target {signal_time_label}", setup["profit_target"], "#66bb6a"),
        ):
            fig.add_hline(
                y=value,
                line_color=color,
                line_dash="dash",
                annotation_text=f"{name}: {value:.5f}",
                annotation_position="top left",
            )

    if sneaky_setups:
        sneaky_rows = [
            analysis.iloc[setup["signal_index"]] for setup in sneaky_setups
        ]
        fig.add_trace(go.Scatter(
            x=[row["Time"] for row in sneaky_rows],
            y=[setup["entry"] for setup in sneaky_setups],
            mode="markers+text",
            text=[f"SP {setup['direction']}" for setup in sneaky_setups],
            textposition=[
                "top center" if setup["direction"] == "BUY" else "bottom center"
                for setup in sneaky_setups
            ],
            marker={
                "symbol": [
                    "triangle-up" if setup["direction"] == "BUY" else "triangle-down"
                    for setup in sneaky_setups
                ],
                "size": 11,
                "color": [
                    "#26c6da" if setup["direction"] == "BUY" else "#ff8a65"
                    for setup in sneaky_setups
                ],
            },
            name="Sneaky Pivot setups",
            customdata=[
                [setup["stop_loss"], setup["profit_target"], setup["level_name"]]
                for setup in sneaky_setups
            ],
            hovertemplate=(
                "SP %{text} · entry %{y:.5f}<br>Stop: %{customdata[0]:.5f}"
                "<br>Target: %{customdata[1]:.5f}<br>Swept: %{customdata[2]}"
                "<extra></extra>"
            ),
        ))
        latest_sneaky = sneaky_setups[-1]
        for name, value, color in (
            ("SP Entry", latest_sneaky["entry"], "#26c6da"),
            ("SP Stop", latest_sneaky["stop_loss"], "#ff8a65"),
            ("SP Target", latest_sneaky["profit_target"], "#66bb6a"),
        ):
            fig.add_hline(
                y=value,
                line_color=color,
                line_dash="dot",
                annotation_text=f"{name}: {value:.5f}",
                annotation_position="bottom right",
            )

    fig.update_layout(
        template="plotly_dark",
        height=560,
        xaxis_title="Time",
        yaxis_title="Price",
        dragmode="pan",
        uirevision=f"{asset_selection}-{timeframe_selection}",
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        margin={"l": 15, "r": 15, "t": 25, "b": 15},
        legend={"orientation": "h", "y": 1.02, "x": 0},
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(128, 132, 149, 0.25)",
        showline=True,
        ticks="outside",
        tickformat="%H:%M\n%b %d",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="#808495",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(128, 132, 149, 0.25)",
        showline=True,
        ticks="outside",
        tickformat=",.2f" if asset_selection == "Gold (XAUUSD)" else ",.5f",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="#808495",
    )
    st.plotly_chart(
        fig,
        width="stretch",
        key=f"price-chart-{asset_selection}-{timeframe_selection}",
        config={
            "displaylogo": False,
            "scrollZoom": True,
        },
    )
    st.subheader(f"Market Structure: {structure_bias}")
    st.caption(
        "Swing highs/lows require two completed candles on each side. Labels are "
        "drawn on the pivot candle only after those right-side candles close. "
        "BOS/CHoCH markers require a candle close through the latest confirmed swing."
    )
    sweep_events = [event for event in structure_events if event["type"] == "sweep"]
    latest_sweep = sweep_events[-1] if sweep_events else None
    st.subheader(
        f"Liquidity Sweeps: {len(sweep_events)} confirmed"
        + (f" · latest {latest_sweep['label']}" if latest_sweep else "")
    )
    st.caption(
        "A bullish sell-side liquidity (SSL) sweep requires a closed candle to "
        "wick below the latest confirmed swing low and close back above it. A "
        "bearish buy-side liquidity (BSL) sweep does the reverse at the latest "
        "confirmed swing high. Each level is marked once; these are informational, "
        "not entry signals."
    )
    def format_zone_status(zone):
        if zone is None:
            return "none active"
        return (
            f"{zone['strength'].lower()} ({zone['strength_score']}/100), "
            f"{zone['status'].lower()}, {zone['touches']} test"
            f"{'' if zone['touches'] == 1 else 's'} "
            f"({zone['low']:.{price_decimals}f}–{zone['high']:.{price_decimals}f})"
        )

    demand_zone = latest_zones["Demand"]
    supply_zone = latest_zones["Supply"]
    st.subheader(
        "Supply & Demand Zones"
        f" · Demand: {format_zone_status(demand_zone)}"
        f" · Supply: {format_zone_status(supply_zone)}"
    )
    st.caption(
        "A zone forms from the last opposing candle before three consecutive "
        "same-direction candles, each with a body at least 60% of its range; "
        "their displacement must be at least 1.5× the previous 10-candle average "
        "range and close beyond the previous 10-candle structure. The 0–100 "
        "strength grade is a heuristic based on displacement and candle bodies, "
        "not a measured win probability. Touches are counted by distinct visits; "
        "a later close beyond the far edge invalidates a zone, even after a "
        "confirmed retest. Latest active zones are shaded; informational only."
    )
    st.subheader("Sneaky Pivot · 15-minute adaptation")
    if timeframe_selection != "15 Minute":
        st.caption("Available on the 15-minute timeframe only.")
    elif previous_day_high is None or previous_day_low is None:
        st.caption(
            "Previous-day levels are unavailable in the simulator candle history."
        )
    elif sneaky_setups:
        latest_sneaky = sneaky_setups[-1]
        st.caption(
            f"{latest_sneaky['direction']} · {latest_sneaky['level_name']} swept · "
            f"confirmed at {pd.Timestamp(latest_sneaky['signal_time']):%Y-%m-%d %H:%M:%S}."
        )
        sneaky_col1, sneaky_col2, sneaky_col3, sneaky_col4 = st.columns(4)
        sneaky_col1.metric("Sneaky Pivot Entry", f"{latest_sneaky['entry']:.{price_decimals}f}")
        sneaky_col2.metric("Stop Loss", f"{latest_sneaky['stop_loss']:.{price_decimals}f}")
        sneaky_col3.metric("Take Profit", f"{latest_sneaky['profit_target']:.{price_decimals}f}")
        sneaky_col4.metric(
            "Previous-Day Range",
            f"{previous_day_low:.{price_decimals}f} – {previous_day_high:.{price_decimals}f}",
        )
    else:
        st.caption(
            f"No confirmed Sneaky Pivot setup in this 15-minute history. "
            f"Previous-day high {previous_day_high:.{price_decimals}f} · "
            f"low {previous_day_low:.{price_decimals}f}."
        )
    if setups:
        setup_history = pd.DataFrame([
            {
                "Confirmed At": pd.Timestamp(setup["signal_time"]).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "Signal": setup["direction"],
                "Confirmation": setup["confirmation_status"],
                "Entry": setup["entry"],
                "Stop Loss": setup["stop_loss"],
                "Take Profit": setup["profit_target"],
                "Chart Status": setup["status"],
            }
            for setup in reversed(setups)
        ])
        st.subheader("Confirmed Setup History")
        st.caption(
            "Each setup keeps its own levels. Chart status is based on later candle "
            "highs/lows only; it is not a tracked broker position. If a candle "
            "touches both levels, their order cannot be determined from OHLC data."
        )
        st.dataframe(
            setup_history,
            hide_index=True,
            width="stretch",
            height=440,
            column_config={
                "Confirmed At": st.column_config.TextColumn(
                    "Confirmed At",
                    width="medium",
                ),
                "Signal": st.column_config.TextColumn(
                    "Signal",
                    width="small",
                ),
                "Confirmation": st.column_config.TextColumn(
                    "Confirmation",
                    width="large",
                ),
                "Entry": st.column_config.NumberColumn(
                    "Entry",
                    format=f"%.{price_decimals}f",
                    width="medium",
                ),
                "Stop Loss": st.column_config.NumberColumn(
                    "Stop Loss",
                    format=f"%.{price_decimals}f",
                    width="medium",
                ),
                "Take Profit": st.column_config.NumberColumn(
                    "Take Profit",
                    format=f"%.{price_decimals}f",
                    width="medium",
                ),
                "Chart Status": st.column_config.TextColumn(
                    "Chart Status",
                    width="large",
                ),
            },
        )
    else:
        st.info("No confirmed breakout setups in the simulator candles.")
    ruler_col1, ruler_col2, ruler_col3 = st.columns(3)
    ruler_col1.metric(
        "Ruler Start",
        f"{ruler_start_candle['Close']:,.{price_decimals}f}",
    )
    ruler_col2.metric(
        "Ruler End",
        f"{ruler_end_candle['Close']:,.{price_decimals}f}",
    )
    ruler_col3.metric(
        "Price Change",
        f"{ruler_price_change:+,.{price_decimals}f}",
        delta=f"{ruler_intervals} bars apart",
    )

    if levels is not None:
        risk_budget = starting_balance * risk_percentage / 100
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Latest Confirmed Signal",
            levels["direction"],
            help=pd.Timestamp(levels["signal_time"]).strftime(
                "Confirmed at %Y-%m-%d %H:%M:%S"
            ),
        )
        col2.metric("Entry", f"{levels['entry']:.5f}")
        col3.metric("Stop Loss", f"{levels['stop_loss']:.5f}")
        col4.metric("Take Profit", f"{levels['profit_target']:.5f}")
        st.caption(
            f"{levels['confirmation_status']} · "
            f"{pd.Timestamp(levels['signal_time']).strftime('%Y-%m-%d %H:%M:%S')} · "
            f"Chart status: {levels['status']} · Risk budget at selected settings: "
            f"${risk_budget:,.2f} · Target profile: {target_multiple:g}:1 · "
            "Signals are informational and do not place trades."
        )
        signal_key = (
            f"{asset_selection}:{timeframe_selection}:"
            f"{pd.Timestamp(levels['signal_time']).isoformat()}:{levels['direction']}"
        )
        recorded_signals = st.session_state.setdefault("recorded_signals", set())
        if st.button(
            "Record simulated trade receipt",
            disabled=signal_key in recorded_signals,
            help="Records a local paper-trade receipt only. It does not place a broker order.",
        ):
            st.session_state["order_logs"].append({
                "Recorded At": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Signal Time": levels["signal_time"],
                "Asset": asset_selection,
                "Timeframe": timeframe_selection,
                "Direction": levels["direction"],
                "Confirmation": levels["confirmation_status"],
                "Entry": levels["entry"],
                "Stop Loss": levels["stop_loss"],
                "Target": levels["profit_target"],
                "Risk Budget": risk_budget,
                "Mode": "Simulated receipt — no order sent",
            })
            recorded_signals.add(signal_key)
            st.success("Simulated trade receipt recorded.")

render_dashboard()

if st.session_state["order_logs"]:
    st.subheader("Simulated Trade Receipts")
    st.dataframe(pd.DataFrame(st.session_state["order_logs"]), hide_index=True)
