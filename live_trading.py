import pandas as pd
import numpy as np
import joblib
import time
import os
from indicator_calculation import compute_indicators

BAR_FILE = "C:/Users/Franc/repos/TradingAI 2/live_data/bar_data.csv"
SIGNAL_FILE = "C:/Users/Franc/repos/TradingA 2/trading/signal.txt"
EXIT_FILE = "C:/Users/Franc/repos/TradingAI 2/trading/exit.txt"
STATUS_FILE = "C:/Users/Franc/repos/TradingAI 2/trading/status.txt"
MODEL_FILE = "C:/Users/Franc/repos/TradingAI 2/rf_model_0.005_0.1_20250101.pkl"

# BAR_FILE = "./trading/bar_data.csv"
# SIGNAL_FILE = "./trading/signal.txt"
# EXIT_FILE = "./trading/exit.txt"
# STATUS_FILE = "./trading/status.txt"
# MODEL_FILE = "rf_model_0.005_0.1_20250101.pkl"

CONFIDENCE_THRESHOLD = 0.1
TRADE_THRESHOLD = 0.005
TICK_VALUE = 5
SL_ATR_MULT = 1.5
TP_ATR_MULT = 3.0
TRAIL_START_MULT = 2.5
TRAIL_STOP_MULT = 1.0
MAX_CONTRACTS = 1

# === Load model ===
model = joblib.load(MODEL_FILE)
class_labels = list(model.classes_)

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
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, "r") as f:
                content = f.read().strip()
            if "flat" in content.lower():
                active_trade_side = None
                print("📭 Trade closed. Status reset to flat.")
            else:
                print("In a trade, status not flat.")
        except Exception as e:
            print(f"[Status Check] Error reading status file: {e}")

def act_on_model(df):
    global active_trade_side
    df = compute_indicators(df)
    latest = df.iloc[-1:]

    # features = [
    #     'rsi', 'macd', 'ema_9', 'ema_21', 'volume', 'chop_index',
    #     'atr_14', 'vwap_diff', 
    #     'body_pct',  
    #     'upper_wick', 'lower_wick',  # just added
    #     'volume_delta_ema'
    # ]

    features = [
        'rsi', 'macd', 'ema_9', 'ema_21', 'atr_14',
    ]
    
    X_new = latest[features]

    pred = model.predict(X_new)[0]
    probs = model.predict_proba(X_new)[0]
    confidence = max(probs)

    if pred not in [-2, -1, 1, 2] or confidence < CONFIDENCE_THRESHOLD:
        return  # Ignore low-confidence or neutral predictions

    side = "long" if pred == 2 or pred == 1 else "short"
    price = latest['close'].values[0]
    atr = latest['atr_14'].values[0]

    expected_mfe = TP_ATR_MULT * atr
    expected_mae = SL_ATR_MULT * atr
    rr = expected_mfe / (expected_mae + 1e-9)

    if rr < 1.2:
        print(f"⛔ Skipped {side} trade due to low RR ({rr:.2f})")
        return  # Skip low risk-reward trades

    # Calculate SL and TP
    sl_price = price - (SL_ATR_MULT * atr) if side == "long" else price + (SL_ATR_MULT * atr)
    tp_price = price + (TP_ATR_MULT * atr) if side == "long" else price - (TP_ATR_MULT * atr)
    trail_trigger = price + (TRAIL_START_MULT * atr) if side == "long" else price - (TRAIL_START_MULT * atr)
    trail_stop = price + (TRAIL_STOP_MULT * atr) if side == "short" else price - (TRAIL_STOP_MULT * atr)

    if active_trade_side != side:
        # New trade, write full entry and SL/TP
        active_trade_side = side
        with open(SIGNAL_FILE, "w") as f:
            f.write(side)
        with open(EXIT_FILE, "w") as f:
            f.write(f"action: entry\n")
            f.write(f"side: {side}\n")
            f.write(f"price: {price:.2f}\n")
            f.write(f"take_profit: {tp_price:.2f}\n")
            f.write(f"stop_loss: {sl_price:.2f}\n")
            f.write(f"trail_trigger: {trail_trigger:.2f}\n")
            f.write(f"trail_stop: {trail_stop:.2f}\n")
    else:
        # In-trade, maybe just update trailing stop or SL
        with open(EXIT_FILE, "w") as f:
            f.write(f"action: update\n")
            f.write(f"side: {side}\n")
            f.write(f"trail_stop: {trail_stop:.2f}\n")
            f.write(f"new_stop: {sl_price:.2f}\n")

    print(f"[{latest['datetime'].values[0]}] Sent {side} @ {price:.2f} | SL: {sl_price:.2f} | TP: {tp_price:.2f} | TrailTrig: {trail_trigger:.2f} | TrailStop: {trail_stop:.2f} | Conf: {confidence:.2f}")


# === Main Loop ===
print("🔁 Waiting for new bars...")
while True:
    try:
        mtime = os.path.getmtime(BAR_FILE)
        if mtime != last_mtime:
            last_mtime = mtime
            df = read_latest_bar()
            check_trade_status()
            act_on_model(df)
        time.sleep(0.01)
        trim_bar_data(BAR_FILE, max_rows=1000)
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(1)