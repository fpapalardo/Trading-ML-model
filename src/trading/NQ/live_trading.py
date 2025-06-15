import os
import time
import traceback
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
import numpy as np
import requests

from indicator_calculation import compute_all_indicators, session_times
from projectx_connector import ProjectXClient
from config import DATA_DIR

# ── Configuration ─────────────────────────────────────────
API_USERNAME   = "pelt8885"
API_KEY        = "IPgPJSFNYyUJp0LwtiqOAUkXfqsdWkA/v1GXll1Hjjs="
CONTRACT_SEARCH = "NQ"
CONTRACT_ID     = None  # will be discovered

BAR_FILE  = f"{DATA_DIR}/live/NQ/bar_data.csv"
LOOKBACK  = 1000       # Initial bar load
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

def retry_api_call(func, max_tries=5, initial_delay=1, exceptions=(requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError)):
    delay = initial_delay
    for attempt in range(max_tries):
        try:
            return func()
        except exceptions as e:
            print(f"[WARN] API call failed ({e.__class__.__name__}): {e}. Retrying in {delay}s...")
            time.sleep(delay)
            delay = min(delay * 2, 30)  # exponential backoff
    print(f"[ERROR] API call failed after {max_tries} attempts. Raising exception.")
    raise

def last_closed_5min_bar_ny(dt=None):
    "Return the datetime of the last closed 5-min bar (NY time)."
    dt = dt or datetime.now(NY_TZ)
    dt = dt.astimezone(NY_TZ)
    minute = (dt.minute // 5) * 5
    bar = dt.replace(minute=minute, second=0, microsecond=0)
    if bar >= dt:
        bar -= timedelta(minutes=5)
    else:
        bar -= timedelta(minutes=0)
    return bar

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
    # Use the last closed 5-min NY bar as endpoint
    end_ny = last_closed_5min_bar_ny()
    start_ny = end_ny - timedelta(minutes=LOOKBACK * 5 - 5)

    # Retry get_bars on SSL or timeout errors
    backoff = 1
    while True:
        try:
            bars = retry_api_call(lambda: px.get_bars(
                CONTRACT_ID,
                start_ny,
                end_ny,
                unit=2, unit_number=5, limit=LOOKBACK
            ))
            break
        except (requests.exceptions.ReadTimeout, requests.exceptions.SSLError) as e:
            print(f"[seed_bars] Error {e.__class__.__name__}: {e}, retrying in {backoff}s...")
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
    df = pd.DataFrame(bars)
    df['t'] = pd.to_datetime(df['t']).dt.tz_convert(NY_TZ)
    df.rename(
        columns={'t':'datetime','o':'open','h':'high','l':'low','c':'close','v':'volume'},
        inplace=True
    )
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    df.to_csv(BAR_FILE)
    return df

if not os.path.exists(BAR_FILE) or os.stat(BAR_FILE).st_size == 0:
    df_window = seed_bars()
else:
    try:
        df_window = pd.read_csv(
            BAR_FILE,
            parse_dates=['datetime'],
            index_col='datetime'
        )

        # Handle timezones safely:
        if not isinstance(df_window.index, pd.DatetimeIndex):
            df_window.index = pd.to_datetime(df_window.index)

        if df_window.index.tz is None:
            df_window.index = df_window.index.tz_localize("America/New_York")
        else:
            # Only convert if not already NY (safe for repeat runs)
            if str(df_window.index.tz) != "America/New_York":
                df_window.index = df_window.index.tz_convert("America/New_York")

    except pd.errors.EmptyDataError:
        df_window = seed_bars()
    df_window = df_window.tail(LOOKBACK)

def is_new_bar(now: datetime, last: datetime, mins: int) -> bool:
    return last is None or (now - last).total_seconds() >= mins * 60

last_15m, last_1h = None, None
cache_15m, cache_1h = None, None

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

in_trade = False

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

    side = 'Buy' if pred == 1 else 'Sell'
    price = ohlcv['close']
    tp_price = price + (tp_atr_mult * atr if pred == 1 else -tp_atr_mult * atr)
    sl_price = price - (sl_atr_mult * atr if pred == 1 else -sl_atr_mult * atr)

    px.place_order(CONTRACT_ID, side, quantity=1)
    result_oco = px.place_oco_exit(
        CONTRACT_ID,
        quantity=1,
        take_profit=tp_price,
        stop_loss=sl_price
    )
    in_trade = True
    print(f"{now} {side}@{price:.2f}, TP={tp_price:.2f}, SL={sl_price:.2f}, OCO={result_oco}")

print("Starting live-trade loop...")
while True:
    try:
        now_ny = datetime.now(NY_TZ)
        last_bar_ny = last_closed_5min_bar_ny(now_ny)
        next_bar_ny = next_5min_boundary(now_ny)
        sleep_secs = (next_bar_ny - now_ny).total_seconds()
        if sleep_secs <= 0:
            sleep_secs = 1

        print(f"[{now_ny:%H:%M:%S}] Sleeping {sleep_secs:.2f}s until {next_bar_ny:%H:%M:%S}")
        time.sleep(sleep_secs)

        # After sleep, get last closed bar in NY time
        last_bar_ny = last_closed_5min_bar_ny()
        last_bar_utc = last_bar_ny.astimezone(timezone.utc)
        bars = retry_api_call(lambda: px.get_bars(
            CONTRACT_ID,
            last_bar_ny,
            last_bar_ny,
            unit=2, unit_number=5, limit=1
        ))

        if bars:
            last = bars[-1]
            dt = pd.to_datetime(last['t']).tz_convert(NY_TZ)
            df_window.sort_index(inplace=True)
            latest_ts = df_window.index.max() if not df_window.empty else None
            if latest_ts is None or dt > latest_ts:
                print(f"New bar detected at {dt}")
                df_window.loc[dt] = [last['o'], last['h'], last['l'], last['c'], last['v']]
                df_window.sort_index(inplace=True)
                pd.DataFrame([{
                    'datetime': dt, 'open': last['o'], 'high': last['h'],
                    'low': last['l'], 'close': last['c'], 'volume': last['v']
                }]).to_csv(
                    BAR_FILE, mode='a', header=False, index=False
                )
                df_window = df_window.tail(LOOKBACK)
                act_on_signal(df_window)
    except Exception:
        print(f"[{datetime.now(timezone.utc)}] Exception:")
        traceback.print_exc()
        time.sleep(5)
