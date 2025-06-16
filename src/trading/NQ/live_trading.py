import os
import time
import traceback
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
import numpy as np
import requests
from requests.exceptions import ReadTimeout, ConnectionError, HTTPError, SSLError

from indicator_calculation import compute_all_indicators, session_times
from projectx_connector import ProjectXClient
from config import DATA_DIR, API_USERNAME, API_KEY

# ── Configuration ─────────────────────────────────────────
CONTRACT_SEARCH = "NQ"
CONTRACT_ID     = None  # will be discovered

BAR_FILE  = f"{DATA_DIR}/live/NQ/bar_data.csv"
LOOKBACK  = 350       # Initial bar load
POLL_SEC   = 30        # Poll interval in seconds

tp_atr_mult = 2.0
sl_atr_mult = 1.5

MODEL_FILE = "rf_model_classifier_LOOKAHEAD_6_session_less.pkl"
FEATURE_COLUMNS = [
    'POC_Dist_Current_Points_1h', 'POC_Dist_Current_Points_5min', 'Day_of_Week',
    'POC_Dist_Current_Points_15min', 'Day_Sin', 'RSI_7_5min', 'Minus_DI_14_1h',
    'Trend_Score_15min', 'Trend_Strength_5min', 'Prev_Swing_Dist_15min',
    'Time_Sin', 'Volume_Trend_15min', 'Is_Trending_5min', 'Is_Trending_15min'
]

# Timezone
NY_TZ = ZoneInfo("America/New_York")

model = joblib.load(MODEL_FILE)

px = ProjectXClient(API_USERNAME, API_KEY)
px.authenticate()

if CONTRACT_ID is None:
    ctrs = px.search_contracts(CONTRACT_SEARCH)
    if not ctrs:
        raise RuntimeError(f"No contract found for '{CONTRACT_SEARCH}'")
    CONTRACT_ID = ctrs[0]['id']

open_orders = px.search_open_orders()
in_trade = len(open_orders) > 0

last_15m, last_1h = None, None
cache_15m, cache_1h = None, None
in_trade = False

def retry_api_call(func, max_tries=5, initial_delay=1):
    """Retry wrapper with exponential backoff and handling of rate limits."""
    delay = initial_delay
    for attempt in range(1, max_tries + 1):
        try:
            return func()
        except HTTPError as e:
            code = e.response.status_code if e.response is not None else None
            if code == 429:
                retry_after = e.response.headers.get('Retry-After')
                sleep_for = int(retry_after) if retry_after and retry_after.isdigit() else delay
                print(f"[WARN] Rate limited (429). Sleeping {sleep_for}s...")
                time.sleep(sleep_for)
                delay = min(delay * 2, 300)
                continue
            print(f"[WARN] HTTP {code} error: {e}. Retrying in {delay}s...")
        except (ReadTimeout, ConnectionError, SSLError) as e:
            print(f"[WARN] Network error ({type(e).__name__}): {e}. Retrying in {delay}s...")
        except Exception as e:
            print(f"[WARN] Unexpected error ({type(e).__name__}): {e}. Retrying in {delay}s...")
        time.sleep(delay)
        delay = min(delay * 2, 300)
    raise RuntimeError(f"API call failed after {max_tries} attempts.")

def last_closed_5min_bar_ny(dt=None):
    dt = dt or datetime.now(NY_TZ)
    dt = dt.astimezone(NY_TZ)
    minute = (dt.minute // 5) * 5
    bar = dt.replace(minute=minute, second=0, microsecond=0)
    if bar >= dt:
        bar -= timedelta(minutes=5)
    return bar

def retry_api_call(func, max_tries=5, initial_delay=1):
    """Retry wrapper for API calls, with exponential backoff and rate-limit handling."""
    delay = initial_delay
    for attempt in range(1, max_tries + 1):
        try:
            return func()
        except HTTPError as e:
            code = e.response.status_code if e.response is not None else None
            if code == 429:
                print(f"[WARN] Rate limited (HTTP 429). Backing off for {delay}s...")
            else:
                print(f"[WARN] HTTP error {code}: {e}. Retrying in {delay}s...")
        except (ReadTimeout, ConnectionError, SSLError) as e:
            print(f"[WARN] Network error ({type(e).__name__}): {e}. Retrying in {delay}s...")
        except Exception as e:
            print(f"[WARN] Unexpected error ({type(e).__name__}): {e}. Retrying in {delay}s...")
        time.sleep(delay)
        delay = min(delay * 2, 60)
    raise RuntimeError(f"API call failed after {max_tries} attempts.")

def load_bars() -> pd.DataFrame:
    # Full seed if no file or empty
    if not os.path.exists(BAR_FILE) or os.stat(BAR_FILE).st_size == 0:
        return seed_bars()
    # Try loading CSV
    try:
        df = pd.read_csv(
            BAR_FILE,
            parse_dates=['datetime'],
            index_col='datetime'
        )
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        print(f"[WARN] CSV load error ({e}); seeding full history")
        return seed_bars()
    # Normalize timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize(NY_TZ)
    else:
        df.index = df.index.tz_convert(NY_TZ)
    # If empty after load, seed
    if df.empty:
        print("[WARN] Loaded CSV empty; seeding full history")
        return seed_bars()
    # Keep only last LOOKBACK bars
    df = df.tail(LOOKBACK)
    # Fetch any missing bars
    last_ts = df.index.max()
    end_ny = last_closed_5min_bar_ny()
    if last_ts < end_ny:
        start_utc = last_ts.astimezone(timezone.utc)
        end_utc   = end_ny.astimezone(timezone.utc)
        bars = retry_api_call(lambda: px.get_bars(
            CONTRACT_ID,
            start_utc,
            end_utc,
            unit=2, unit_number=5,
            limit=int((end_ny - last_ts).total_seconds() // 300)
        ))
        if bars:
            new_df = pd.DataFrame(bars)
            new_df['t'] = pd.to_datetime(new_df['t'], utc=True).dt.tz_convert(NY_TZ)
            new_df.rename(columns={
                't':'datetime','o':'open','h':'high',
                'l':'low','c':'close','v':'volume'
            }, inplace=True)
            new_df.set_index('datetime', inplace=True)
            new_df.sort_index(inplace=True)
            new_df = new_df[new_df.index > last_ts]
            if not new_df.empty:
                df = pd.concat([df, new_df]).tail(LOOKBACK)
                new_df.to_csv(BAR_FILE, mode='a', header=False)
    return df

def next_5min_boundary(dt: datetime) -> datetime:
    """
    Given a tz-aware datetime `dt`, return the next datetime
    whose minute % 5 == 0, second == 0, microsecond == 0.
    """
    # how many minutes past the last 5-min mark?
    over = dt.minute % 5
    # minutes to add to hit the next multiple of 5
    to_add = (5 - over) if over != 0 else 5
    # zero out seconds/microseconds and add
    next_bar = (dt
        .replace(second=0, microsecond=0)
        + timedelta(minutes=to_add)
    )
    return next_bar

def seed_bars() -> pd.DataFrame:
    end_ny = last_closed_5min_bar_ny()
    start_ny = end_ny - timedelta(minutes=LOOKBACK * 5 - 5)
    bars = retry_api_call(lambda: px.get_bars(
        CONTRACT_ID,
        start_ny.astimezone(timezone.utc),
        end_ny.astimezone(timezone.utc),
        unit=2, unit_number=5,
        limit=LOOKBACK
    ))
    df = pd.DataFrame(bars)
    df['t'] = pd.to_datetime(df['t'], utc=True).dt.tz_convert(NY_TZ)
    df.rename(columns={
        't':'datetime','o':'open','h':'high',
        'l':'low','c':'close','v':'volume'
    }, inplace=True)
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    df.to_csv(BAR_FILE)
    return df

def is_new_bar(now: datetime, last: datetime, mins: int) -> bool:
    return last is None or (now - last).total_seconds() >= mins * 60

def prepare_features(df5: pd.DataFrame,
                     df15: pd.DataFrame,
                     df1h: pd.DataFrame,
                     now: datetime) -> pd.DataFrame:
    f5 = compute_all_indicators(df5.copy(), suffix='_5min', features=['all'])
    f5 = session_times(f5)

    global last_15m, cache_15m
    if is_new_bar(now, last_15m, 15):
        cache_15m = compute_all_indicators(df15.copy(), suffix='_15min', features=['volume_trend', 'prev_swing', 'trend', 'poc', 'adx', 'ema', 'atr'])
        last_15m = now
    f15 = cache_15m

    global last_1h, cache_1h
    if is_new_bar(now, last_1h, 60):
        cache_1h = compute_all_indicators(df1h.copy(), suffix='_1h', features=['adx', 'poc'])
        last_1h = now
    f1h = cache_1h

    df = pd.merge_asof(
        f5.sort_index(), f15.filter(regex='_15min$').sort_index(),
        left_index=True, right_index=True, direction='backward'
    )
    df = pd.merge_asof(
        df.sort_index(), f1h.filter(regex='_1h$').sort_index(),
        left_index=True, right_index=True, direction='backward'
    )
    df.ffill(inplace=True)
    return df.tail(1)

def act_on_signal(df: pd.DataFrame):
    global in_trade
    if in_trade:
        print("[act_on_signal] In trade, skipping signal logic.")
        return

    if df.empty:
        print("[act_on_signal] ERROR: DataFrame is empty, cannot resample.")
        return

    # Clean and check column names
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace('\ufeff', '')
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            print(f"[act_on_signal] ERROR: Missing column {col}! Columns: {list(df.columns)}")
            return

    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }

    # Resample 15m
    try:
        df15 = df.resample('15min', closed='left', label='left').agg(agg_dict).dropna()
    except Exception as e:
        print("[act_on_signal] ERROR during df15 resample:", e)
        print("[act_on_signal] Columns at resample:", list(df.columns))
        return

    # Resample 1h
    try:
        df1h = df.resample('1h', closed='left', label='left').agg(agg_dict).dropna()
    except Exception as e:
        print("[act_on_signal] ERROR during df1h resample:", e)
        print("[act_on_signal] Columns at resample:", list(df.columns))
        return

    # Proceed with original logic
    now = df.index[-1]
    ohlcv = df.iloc[-1][['open','high','low','close','volume']]

    feats = prepare_features(df, df15, df1h, now)
    if feats.empty:
        print("[act_on_signal] Feature DataFrame is empty after prepare_features.")
        return

    atr = feats['ATR_14_5min'].iat[0]
    X = feats[FEATURE_COLUMNS]
    pred = model.predict(X)[0]
    if pred not in (1, 2):
        print("No Trade prediction, waiting for next candle")
        return

    open_side = 'Buy' if pred == 1 else 'Sell'
    exit_side = 'Sell' if open_side == 'Buy' else 'Buy'

    print(f"Predicted {open_side}")
    price = ohlcv['close']
    tp_price = price + (tp_atr_mult * atr if pred == 1 else -tp_atr_mult * atr)
    sl_price = price - (sl_atr_mult * atr if pred == 1 else -sl_atr_mult * atr)

    px.place_order(CONTRACT_ID, open_side, quantity=1)
    result_oco = px.place_oco_exit(
        CONTRACT_ID,
        quantity=1,
        take_profit=tp_price,
        stop_loss=sl_price,
        side=exit_side
    )
    in_trade = True
    print(f"{now} {open_side}@{price:.2f}, TP={tp_price:.2f}, SL={sl_price:.2f}, OCO={result_oco}")

try:
    df_window = load_bars()
except Exception as e:
    print(f"[ERROR] Initial load_bars failed: {e}")
    df_window = pd.DataFrame(columns=['open','high','low','close','volume'])

prev_expected_ts = None
# Main loop
print("Starting live-trade loop...")
while True:
    try:
        now_ny   = datetime.now(NY_TZ)
        expected_bar = last_closed_5min_bar_ny(now_ny)
        next_bar = next_5min_boundary(now_ny)
        sleep_s  = max((next_bar - now_ny).total_seconds(), 1)
        print(f"[{now_ny:%H:%M:%S}] Sleeping {sleep_s:.1f}s until {next_bar:%H:%M:%S}")
        time.sleep(sleep_s)
        # Reload bars (seed if needed)
        df_window = load_bars()

        # Only run if expected bar is present and not yet processed
        if expected_bar in df_window.index and expected_bar != prev_expected_ts:
            print(f"Expected bar detected: {expected_bar}. Running trading logic.")
            act_on_signal(df_window)
            prev_expected_ts = expected_bar
        else:
            print(f"Skipping: expected bar {expected_bar} not ready or already processed.")

    except Exception:
        print(f"[{datetime.now(timezone.utc)}] Exception:")
        traceback.print_exc()
        time.sleep(5)
