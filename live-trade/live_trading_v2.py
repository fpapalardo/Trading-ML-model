import pandas as pd
import numpy as np
import joblib
import time
import datetime
import os
import io
from indicator_calculation import compute_indicators
import traceback
from collections import deque

BAR_FILE = "./live_data/bar_data.csv"
SIGNAL_FILE = "./trading/signal.txt"
EXIT_FILE = "./trading/exit.txt"
STATUS_FILE = "./trading/status.txt"

TRADE_THRESHOLD = 0.001
TICK_VALUE = 5
SL_ATR_MULT = 1.5
TP_ATR_MULT = 2.0
TRAIL_START_MULT = 1.0
TRAIL_STOP_MULT = 0.5

# Order size
BASE_CONTRACTS = 1
MAX_CONTRACTS = 3
PRED_HISTORY = deque(maxlen=100)

# === Load model ===
model = joblib.load("stack_model_LOOKAHEAD_5_session_less.pkl")

# === Track order info ===
entry_price = None
trail_trigger = None

# === Track last file modification time ===
last_mtime = None
active_trade_side = None  # Track active trade

# === Load initial window ===
df_window = pd.read_csv(BAR_FILE, names=["datetime", "open", "high", "low", "close", "volume"])
df_window['datetime'] = pd.to_datetime(df_window['datetime'])
df_window = df_window.tail(100).copy()

def check_trade_status():
    global active_trade_side, entry_price, trail_trigger
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, "r") as f:
                content = f.read().strip()
            if "flat" in content.lower():
                active_trade_side = None
                entry_price = None
                trail_trigger = None
                return True
            else:
                return False
        except Exception as e:
            print(f"[Status Check] Error reading status file: {e}")

def wait_for_next_minute():
    """Wait until the next full minute plus 0.5 seconds (e.g., 08:01:00.500)."""
    now = datetime.datetime.now()
    next_minute = (now + datetime.timedelta(minutes=1)).replace(second=0, microsecond=0)
    target_time = next_minute + datetime.timedelta(seconds=0.1)
    sleep_duration = (target_time - now).total_seconds()
    if sleep_duration > 0:
        time.sleep(sleep_duration)

def act_on_model(df):
    global active_trade_side, trail_trigger, entry_price

    df = compute_indicators(df)
    if df.shape[0] < 13:
        print("⏭️ Skipping: Not enough data (requires at least 13 rows).")
        return
    
    latest = df.iloc[-1]
    is_flat = check_trade_status()

    features = ["atr_pct", "atr_5", "rsi_6", "ema_dist", "macd_fast", 
     "macd_fast_diff", "is_pivot_low_15", 
     "is_pivot_high_5", "is_pivot_low_5", 
     "is_pivot_high_10", "is_pivot_low_10", "is_pivot_high_15"]

    if is_flat:
        X_new = df.iloc[[-1]][features]
        pred = model.predict(X_new)[0]

        # === Store pred for confidence scaling ===
        PRED_HISTORY.append(pred)

        # === Calculate confidence z-score
        if len(PRED_HISTORY) >= 10:
            preds_array = np.array(PRED_HISTORY)
            zscore = (pred - preds_array.mean()) / (preds_array.std() + 1e-9)
            conf = np.clip(abs(zscore), 0, 2.0)
            position_size = BASE_CONTRACTS + (MAX_CONTRACTS - BASE_CONTRACTS) * (conf / 2.0)
            position_size = round(position_size, 2)
        else:
            conf = 0
            position_size = BASE_CONTRACTS  # fallback until history fills

        # === Trade direction based on threshold ===
        if pred  >= TRADE_THRESHOLD:
            side = "long"
        elif pred  <= -TRADE_THRESHOLD:
            side = "short"
        else:
            print(f"⏭️ Skipping: Prediction {pred :.4f} not strong enough.")
            return  # prediction not strong enough

        latest = df.iloc[-1]

        entry_price = latest['close']
        atr = latest['atr_5']

        # === Calculate SL and TP using fixed ATR multipliers ===
        sl_price = entry_price - SL_ATR_MULT * atr if side == "long" else entry_price + SL_ATR_MULT * atr
        expected_move = abs(pred) * entry_price
        min_tp = 0.001 * entry_price
        max_tp = TP_ATR_MULT * atr
        tp_move = np.clip(expected_move, min_tp, max_tp)
        tp_price = entry_price + tp_move if side == "long" else entry_price - tp_move
        trail_trigger = entry_price + TRAIL_START_MULT * atr if side == "long" else entry_price - TRAIL_START_MULT * atr

        active_trade_side = side
    
        # New trade, write full entry and SL/TP
        active_trade_side = side
        with open(SIGNAL_FILE, "w") as f:
            f.write(f"action: entry\n")
            f.write(f"side: {side}\n")
            f.write(f"price: {entry_price:.2f}\n")
            f.write(f"take_profit: {tp_price:.2f}\n")
            f.write(f"stop_loss: {sl_price:.2f}\n")      
            f.write(f"trail_trigger: {trail_trigger:.2f}\n")
            f.write(f"size: {position_size:.2f}\n")

        with open(STATUS_FILE, "w") as f:
            f.write(f"In Trade")  

        print(f"[{latest['datetime']}] 🚀 ENTRY: {side.upper()} | Size: {position_size} | Conf: {conf:.2f}")
        print(f"[{latest['datetime']}] 🚀 ENTRY: {side.upper()} @ {entry_price:.2f} | SL: {sl_price:.2f} | TP: {tp_price:.2f} | TrailTrig: {trail_trigger:.2f}")
    else:
        atr = latest['atr_5']
        trail_stop = None

        if active_trade_side == 'long' and latest['high'] >= trail_trigger:
            trail_stop = latest['close'] - TRAIL_STOP_MULT * atr
        elif active_trade_side == 'short' and latest['low'] <= trail_trigger:
            trail_stop = latest['close'] + TRAIL_STOP_MULT * atr

        if trail_stop is not None:
            with open(EXIT_FILE, "w") as f:
                f.write(f"action: update\n")
                f.write(f"side: {active_trade_side}\n")
                f.write(f"trail_stop: {trail_stop:.2f}\n")
            print(f"[{latest['datetime']}] 🛑 Updated | TrailStop: {trail_stop:.2f}")
        else:
            print(f"[{latest['datetime']}] No update needed | Price: {latest['close']:.2f}")

def process_new_bar(bar_df):
    global df_window
    df_window = pd.concat([df_window, bar_df], ignore_index=True).tail(100)
    act_on_model(df_window)

def tail_csv(file_path, callback, sleep_interval=0.02):
    with open(file_path, 'r') as f:
        f.seek(0, os.SEEK_END)  # Go to end of file

        while True:
            line = f.readline()
            if not line:
                time.sleep(sleep_interval)
                continue

            try:
                row_df = pd.read_csv(io.StringIO(line), header=None)
                row_df.columns = ["datetime", "open", "high", "low", "close", "volume"]
                row_df['datetime'] = pd.to_datetime(row_df['datetime'])
                callback(row_df)
            except Exception as e:
                print(f"⚠️ Failed to process line: {line.strip()} | Error: {e}")
                traceback.print_exc()

# === Main Loop ===
print("🔁 Tailing bar_data.csv for new bars...")
tail_csv(BAR_FILE, process_new_bar)