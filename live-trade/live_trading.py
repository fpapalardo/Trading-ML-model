import pandas as pd
import numpy as np
import joblib
import time
import datetime
import os
from indicator_calculation import compute_indicators

BAR_FILE = "./live_data/bar_data.csv"
SIGNAL_FILE = "./trading/signal.txt"
EXIT_FILE = "./trading/exit.txt"
STATUS_FILE = "./trading/status.txt"

# BAR_FILE = "./trading/bar_data.csv"
# SIGNAL_FILE = "./trading/signal.txt"
# EXIT_FILE = "./trading/exit.txt"
# STATUS_FILE = "./trading/status.txt"
# MODEL_FILE = "rf_model_0.005_0.1_20250101.pkl"

TRADE_THRESHOLD = 0.001
TICK_VALUE = 5
SL_ATR_MULT = 1.5
TP_ATR_MULT = 3.0
TRAIL_START_MULT = 2.5
TRAIL_STOP_MULT = 1.0
MAX_CONTRACTS = 1

# === Load model ===
scaler = joblib.load("scaler_LOOKAHEAD_5.pkl")
model = joblib.load("stack_model_LOOKAHEAD_5.pkl")

# === Track order info ===
entry_price = None
trail_trigger = None

# === Track last file modification time ===
last_mtime = None
active_trade_side = None  # Track active trade

def read_latest_bar():
    df = pd.read_csv(BAR_FILE, names=["datetime", "open", "high", "low", "close", "volume"])
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df.tail(100)

def trim_bar_data(file_path: str, max_rows: int = 1000):
    try:
        df = pd.read_csv(file_path)
        if len(df) > max_rows:
            df.tail(max_rows).to_csv(file_path, index=False)
    except Exception as e:
        print(f"[Cleanup] Failed to trim bar data: {e}")

def check_trade_status():
    global active_trade_side
    global entry_price
    global trail_trigger
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, "r") as f:
                content = f.read().strip()
            if "flat" in content.lower():
                active_trade_side = None
                entry_price = None
                trail_trigger = None
                print("📭 Currently Flat")
                return True
            else:
                print("In a trade, status not flat.")
                return False
        except Exception as e:
            print(f"[Status Check] Error reading status file: {e}")

def wait_for_next_minute():
    """Wait until the next full minute begins (e.g., 08:01:00.000)."""
    now = datetime.datetime.now()
    seconds_until_next_minute = 60 - now.second - now.microsecond / 1_000_000
    time.sleep(seconds_until_next_minute)

def act_on_model(df):
    global active_trade_side
    global trail_trigger
    global entry_price

    df = compute_indicators(df)
    latest = df.iloc[-1:]
    if df.shape[0] < 50:
        print("⏭️ Skipping: Not enough data (requires at least 100 rows).")
        return

    features = [
        'atr_5',
        'is_pivot_low_5',
        'atr_pct',
        'rsi_6',
        'macd_fast_diff',
        'is_pivot_high_5',
        'candle_range',
        'return_1',
        'macd_fast',
        'volume',
        'is_pivot_low_10',
        'is_pivot_high_10'
    ]

    X_new = latest[features]  # Convert to NumPy
    X_new_scaled = scaler.transform(X_new)  # Scale like training

    # === Predict return from regression model ===
    pred_return = model.predict(X_new_scaled)[0]

    # === Trade direction based on threshold ===
    if pred_return >= TRADE_THRESHOLD:
        side = "long"
        print(f"📈 Prediction: {pred_return:.4f} | Side: {side}")
    elif pred_return <= -TRADE_THRESHOLD:
        side = "short"
        print(f"📉 Prediction: {pred_return:.4f} | Side: {side}")
    else:
        print(f"⏭️ Skipping: Prediction {pred_return:.4f} not strong enough.")
        return  # prediction not strong enough

    latest = df.iloc[-1]
    
    if check_trade_status():
        entry_price = latest['close']
        atr = latest['atr_14']

        # === Calculate SL and TP using fixed ATR multipliers ===
        sl_price = entry_price - SL_ATR_MULT * atr if side == "long" else entry_price + SL_ATR_MULT * atr
        # Clip TP between 0.1% and TP_ATR_MULT * ATR
        expected_move = abs(pred_return) * entry_price
        min_tp = 0.001 * entry_price
        max_tp = TP_ATR_MULT * atr
        tp_move = np.clip(expected_move, min_tp, max_tp)
        print(f"Expected move: {expected_move:.2f} | Min TP: {min_tp:.2f} | Max TP: {max_tp:.2f}")
        tp_price = entry_price + tp_move if side == "long" else entry_price - tp_move
        trail_trigger = entry_price + TRAIL_START_MULT * atr if side == "long" else entry_price - TRAIL_START_MULT * atr
    
        # New trade, write full entry and SL/TP
        active_trade_side = side
        with open(SIGNAL_FILE, "w") as f:
            f.write(f"action: entry\n")
            f.write(f"side: {side}\n")
            f.write(f"price: {entry_price:.2f}\n")
            f.write(f"take_profit: {tp_price:.2f}\n")
            f.write(f"stop_loss: {sl_price:.2f}\n")      

        with open(STATUS_FILE, "w") as f:
            f.write(f"In Trade")  

        print(f"[{latest['datetime']}] 🚀 Entry: {side.upper()} @ {entry_price:.2f} | SL: {sl_price:.2f} | TP: {tp_price:.2f} | TrailTrig: {trail_trigger:.2f}")
    
    elif not check_trade_status():
        if active_trade_side == 'long' and latest['high'] >= trail_trigger:
            trail_stop = latest['close'] - TRAIL_STOP_MULT * atr
        elif active_trade_side == 'short' and latest['low'] <= trail_trigger:
            trail_stop = latest['close'] + TRAIL_STOP_MULT * atr
        else:
            trail_stop = None

        if trail_stop is not None:
            with open(EXIT_FILE, "w") as f:
                f.write(f"action: update\n")
                f.write(f"side: {active_trade_side}\n")
                f.write(f"trail_stop: {trail_stop:.2f}\n")

            print(f"[{latest['datetime']}] Updated | TrailStop: {trail_stop:.2f}")
        else:
            print(f"[{latest['datetime']}] No update needed | Current price: {latest['close']:.2f}")

# === Main Loop ===
print("🔁 Waiting for new bars...")
while True:
    try:
        # Wait until the *next* full minute again
        wait_for_next_minute()
        df = read_latest_bar()
        act_on_model(df)
        trim_bar_data(BAR_FILE, max_rows=1000)
        
    except Exception as e:
        print(f"Error: {e}")