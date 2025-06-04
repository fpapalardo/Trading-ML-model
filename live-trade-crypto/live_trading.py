import os
import time
import pandas as pd
from datetime import datetime, timezone
from binance.client import Client
from binance.exceptions import BinanceAPIException
from indicator_calculation import compute_all_indicators, add_time_session_features
import joblib
import numpy as np
from collections import deque

# === Setup Binance Client ===
api_key = "9pK67NsdYya1DoX8HJpeEiv93cg8GRRK2b4mTD43CZZCr5H5cUJSNkmlHSBLfL2t"
api_secret = "qfBeNTseuTqQ9YL0qKy6udtnysUimKwfPlqPhUd0lJBXFFxtwMojr2CP40BhmsKH"
client = Client(api_key, api_secret)

# === Configurations ===
SYMBOL = "BTCUSDT"
INTERVALS = {
    "5min": Client.KLINE_INTERVAL_5MINUTE,
    "15min": Client.KLINE_INTERVAL_15MINUTE,
    "1h": Client.KLINE_INTERVAL_1HOUR
}

TRADE_THRESHOLD = 0.000004
SL_ATR_MULT = 0.1
TP_ATR_MULT = 2.0
TRAIL_START_MULT = 1.0
TRAIL_STOP_MULT = 0.3

BASE_CONTRACTS = 1
MAX_CONTRACTS = 1
PRED_HISTORY = deque(maxlen=1000)

model = joblib.load("catboost_model_regression_BTC-12&24-single-model-less-features.pkl")

entry_price = None
trail_trigger = None
active_trade_side = None
trail_stop_price = None
last_sl_order_id = None
open_order_ids = []

last_5m_time = None
last_df5 = pd.DataFrame()
df15_cached = pd.DataFrame()
df1h_cached = pd.DataFrame()
last_15min_pull = None
last_1h_pull = None

model_features = [

   # === 5min Features ===
    "EMA_9_5min", "EMA_21_5min",
    "MACD_12_26_9_5min", "MACDh_12_26_9_5min",
    "RSI_14_5min", "ADX_14_5min", "ATR_14_5min",
    "Volume_SMA_20_5min",
    "VWAP_D_5min", "close_vs_VWAP_D_5min",
    "DCU_20_20_5min", "DCL_20_20_5min",
    "close_vs_EMA20_5min", "EMA_9_vs_EMA21_5min",
    "CHOP_14_1_100_5min",

    # === 15min Features ===
    "EMA_20_15min", "EMA_50_15min",
    "MACD_12_26_9_15min", "MACDh_12_26_9_15min",
    "RSI_14_15min", "ADX_14_15min", "ATR_14_15min",
    "VWAP_D_15min", "DCU_20_20_15min", "DCL_20_20_15min",
    "close_vs_EMA20_15min",

    # === 1h Features ===
    "EMA_50_1h", "EMA_200_1h",
    "MACD_12_26_9_1h", "MACDh_12_26_9_1h",
    "RSI_14_1h", "ADX_14_1h", "ATR_14_1h",
    "VWAP_D_1h", "CHOP_14_1_100_1h",
    "DCU_20_20_1h", "DCL_20_20_1h",
    "close_vs_EMA200_1h"
]

def fetch_last_1000_candles(symbol, interval):
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=1000)
        df = pd.DataFrame([{
            'datetime': datetime.fromtimestamp(k[0] / 1000, timezone.utc),
            'open': float(k[1]), 'high': float(k[2]),
            'low': float(k[3]), 'close': float(k[4]), 'volume': float(k[5])
        } for k in klines])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index("datetime", inplace=True)
        return df
    except Exception as e:
        print(f"❌ Error fetching candles: {e}")
        return pd.DataFrame()

def check_trade_status():
    global active_trade_side, entry_price, trail_trigger, open_order_ids
    try:
        positions = client.get_margin_account()['userAssets']
        btc_position = next((p for p in positions if p['asset'] == 'BTC'), None)
        if btc_position and (float(btc_position['borrowed']) > 0 or float(btc_position['free']) > 0):
            return False
        # Reset if no active position
        active_trade_side = None
        entry_price = None
        trail_trigger = None
        open_order_ids.clear()
        return True
    except Exception as e:
        print(f"[Status Check] Binance API error: {e}")
        return False

def get_wallet_balance(asset="USDT"):
    try:
        margin_data = client.get_isolated_margin_account()
        asset_data = next((a for a in margin_data['assets'] if a['symbol'] == SYMBOL), None)
        if asset_data:
            return float(asset_data['quoteAsset']['free'])
    except Exception as e:
        print(f"❌ Wallet balance fetch error: {e}")
    return 0.0

def calculate_borrow_amount(entry_price, max_leverage=3.0):
    usdt_balance = get_wallet_balance()
    if usdt_balance == 0 or entry_price is None or entry_price <= 0 or not np.isfinite(entry_price):
        print(f"❌ Invalid entry_price or wallet balance → entry_price={entry_price}, balance={usdt_balance}")
        return 0.0, 0.0
    total_usdt = usdt_balance * max_leverage
    borrow_needed = total_usdt - usdt_balance
    quantity = total_usdt / entry_price
    return borrow_needed, quantity


def borrow_margin(asset="USDT", amount=2, symbol="BTCUSDT", is_isolated=True):
    try:
        client.create_margin_loan(asset=asset, amount=str(amount), isIsolated=is_isolated, symbol=symbol)
        print(f"✅ Borrowed {amount} {asset}")
        return True
    except BinanceAPIException as e:
        print(f"❌ Borrow failed: {e}")
        return False

def place_margin_order(symbol="BTCUSDT", side="BUY", quantity=0.001, is_isolated=True):
    try:
        order = client.create_margin_order(symbol=symbol, side=side, type="MARKET",
                                           quantity=quantity, isIsolated=is_isolated)
        print(f"✅ {side} order executed: {order['executedQty']} {symbol.replace('USDT','')}")
    except BinanceAPIException as e:
        print(f"❌ Order failed: {e}")

def place_fake_oco(symbol, side, quantity, tp_price, sl_price):
    global open_order_ids, last_sl_order_id
    try:
        opposite = "SELL" if side == "BUY" else "BUY"

        # TP LIMIT order
        tp_order = client.create_margin_order(
            symbol=symbol, side=opposite, type="LIMIT",
            timeInForce="GTC", quantity=quantity,
            price=str(round(tp_price, 2)), isIsolated=True
        )

        # SL STOP_MARKET
        sl_order = client.create_margin_order(
            symbol=symbol, side=opposite, type="STOP_MARKET",
            stopPrice=str(round(sl_price, 2)),
            quantity=quantity, isIsolated=True
        )

        open_order_ids = [tp_order['orderId'], sl_order['orderId']]
        last_sl_order_id = sl_order['orderId']
        print(f"📌 TP + initial SL placed (fake OCO)")
    except BinanceAPIException as e:
        print(f"❌ Failed to place TP/SL (OCO-style): {e}")

def update_trailing_stop(new_sl_price, quantity, side):
    global last_sl_order_id, trail_stop_price
    try:
        if last_sl_order_id:
            try:
                client.cancel_margin_order(symbol=SYMBOL, orderId=last_sl_order_id, isIsolated=True)
                print(f"🔁 Canceled old SL: {last_sl_order_id}")
            except BinanceAPIException as ce:
                print(f"⚠️ Cancel SL error: {ce}")

        # Place new STOP_MARKET SL
        opposite = "SELL" if side == "BUY" else "BUY"
        sl_order = client.create_margin_order(
            symbol=SYMBOL, side=opposite,
            type="STOP_MARKET",
            stopPrice=str(round(new_sl_price, 2)),
            quantity=quantity,
            isIsolated=True
        )
        last_sl_order_id = sl_order['orderId']
        trail_stop_price = new_sl_price
        print(f"🛡️ Trailing SL updated: {new_sl_price:.2f}")
    except BinanceAPIException as e:
        print(f"❌ Trailing SL error: {e}")


def cancel_remaining_orders():
    global open_order_ids
    try:
        for oid in open_order_ids:
            try:
                client.cancel_margin_order(symbol=SYMBOL, orderId=oid, isIsolated=True)
                print(f"❎ Canceled order {oid}")
            except BinanceAPIException as e:
                if "UNKNOWN_ORDER" in str(e):
                    print(f"🟢 Order {oid} already filled or not found")
        open_order_ids.clear()
    except Exception as e:
        print(f"❌ Order cancel error: {e}")

def check_if_order_filled():
    global open_order_ids
    for oid in open_order_ids:
        try:
            order = client.get_margin_order(symbol=SYMBOL, orderId=oid, isIsolated=True)
            if order["status"] == "FILLED":
                print(f"✅ Order {oid} filled, canceling others...")
                cancel_remaining_orders()
                return True
        except Exception as e:
            print(f"❌ Order status check failed: {e}")
    return False

def act_on_model():
    global active_trade_side, trail_trigger, entry_price
    global df15_cached, df1h_cached, last_15min_pull, last_1h_pull
    global last_5m_time, last_df5

    now = datetime.now(timezone.utc)
    df5 = fetch_last_1000_candles(SYMBOL, INTERVALS["5min"])

    if df5.empty:
        return

    last_candle_time = df5.index[-1]
    if last_5m_time == last_candle_time:
        print("⏭️ No new 5m candle, skipping...")
        return

    last_df5 = df5.copy()
    last_5m_time = last_candle_time
    df5 = compute_all_indicators(df5, suffix="_5min", indicators=["EMA", "MACD", "RSI", "ADX", "ATR", "Volume_SMA", "VWAP", "DC", "SLOPE", "CHOP"])

    if last_15min_pull is None or (now - last_15min_pull).seconds >= 60:
        if now.minute % 15 == 0:
            df15_cached = fetch_last_1000_candles(SYMBOL, INTERVALS["15min"])
            df15_cached = compute_all_indicators(df15_cached, suffix="_15min", indicators=["EMA", "MACD", "RSI", "ADX", "ATR",
                "VWAP", "DC", "SLOPE"])
            last_15min_pull = now
        elif last_15min_pull is None:
            df15_cached = fetch_last_1000_candles(SYMBOL, INTERVALS["15min"])
            df15_cached = compute_all_indicators(df15_cached, suffix="_15min", indicators=["EMA", "MACD", "RSI", "ADX", "ATR",
                "VWAP", "DC", "SLOPE"])
            last_15min_pull = now

    if last_1h_pull is None or (now - last_1h_pull).seconds >= 60:
        if now.minute == 0:
            df1h_cached = fetch_last_1000_candles(SYMBOL, INTERVALS["1h"])
            df1h_cached = compute_all_indicators(df1h_cached, suffix="_1h", indicators=["EMA", "MACD", "RSI", "ADX", "ATR", 
                "VWAP", "CHOP", "DC","SLOPE"])
            last_1h_pull = now
        elif last_1h_pull is None:
            df1h_cached = fetch_last_1000_candles(SYMBOL, INTERVALS["1h"])
            df1h_cached = compute_all_indicators(df1h_cached, suffix="_1h", indicators=["EMA", "MACD", "RSI", "ADX", "ATR", 
                "VWAP", "CHOP", "DC","SLOPE"])
            last_1h_pull = now

    df = df5.copy()
    if not df15_cached.empty:
        df = pd.merge_asof(df.sort_index(), df15_cached.filter(regex="_15min$").sort_index(),
                           left_index=True, right_index=True, direction='backward')
    if not df1h_cached.empty:
        df = pd.merge_asof(df.sort_index(), df1h_cached.filter(regex="_1h$").sort_index(),
                           left_index=True, right_index=True, direction='backward')

    df.ffill(inplace=True)
    if df.empty or not all(f in df.columns for f in model_features):
        print("❌ Missing features or empty data")
        return

    is_flat = check_trade_status()
    last_row = df[model_features].iloc[-1:]
    pred = model.predict(last_row)[0]
    PRED_HISTORY.append(pred)
    
    if is_flat:
        if abs(pred) <= TRADE_THRESHOLD:
            print(f"⏭️ Skip: prediction {pred:.10f} not strong enough")
            return

        side = "long" if pred >= TRADE_THRESHOLD else "short"
        latest = df.iloc[-1]
        entry_price = latest['close']
        atr = latest['ATR_14_5min']
        expected_move = abs(pred) * entry_price
        tp_move = np.clip(expected_move, 0.005 * entry_price, TP_ATR_MULT * atr)
        sl_move = SL_ATR_MULT * atr
        tp_price = entry_price + tp_move if side == "long" else entry_price - tp_move
        sl_price = entry_price - sl_move if side == "long" else entry_price + sl_move
        trail_trigger = entry_price + TRAIL_START_MULT * atr if side == "long" else entry_price - TRAIL_START_MULT * atr
        active_trade_side = side

        borrow_amt, quantity = calculate_borrow_amount(entry_price)

        # === VALIDATION ===
        if any([
            borrow_amt is None, quantity is None,
            borrow_amt <= 0, quantity <= 0,
            not np.isfinite(borrow_amt), not np.isfinite(quantity)
        ]):
            print(f"❌ Invalid borrow_amt or quantity → Borrow: {borrow_amt}, Qty: {quantity}")
            return

        # === Proceed only if valid ===
        borrow_margin(asset="USDT", amount=round(borrow_amt, 2), symbol=SYMBOL)
        time.sleep(2)

        place_margin_order(
            symbol=SYMBOL,
            side="BUY" if side == "long" else "SELL",
            quantity=round(quantity, 6)
        )
        time.sleep(2)

        place_fake_oco(
            SYMBOL,
            "BUY" if side == "long" else "SELL",
            round(quantity, 6),
            tp_price,
            sl_price
        )


        print(f"📈 Entry {side.upper()} @ {entry_price:.2f} | TP: {tp_price:.2f} | SL: {sl_price:.2f} | Trail: {trail_trigger:.2f}")
    else:
        latest = df.iloc[-1]
        atr = latest['ATR_14_5min']
        high, low, close = latest['high'], latest['low'], latest['close']
        quantity = MAX_CONTRACTS  # replace with actual qty logic if needed
        side = "BUY" if active_trade_side == "long" else "SELL"

        if active_trade_side == 'long' and high >= trail_trigger:
            new_sl = close - TRAIL_STOP_MULT * atr
            if trail_stop_price is None or new_sl > trail_stop_price:
                update_trailing_stop(new_sl, quantity, side)

        elif active_trade_side == 'short' and low <= trail_trigger:
            new_sl = close + TRAIL_STOP_MULT * atr
            if trail_stop_price is None or new_sl < trail_stop_price:
                update_trailing_stop(new_sl, quantity, side)

        check_if_order_filled()


# === Run loop ===
while True:
    now = datetime.now(timezone.utc)
    next_minute = (now + pd.Timedelta(minutes=1)).replace(second=0, microsecond=0)
    sleep_seconds = (next_minute - datetime.now(timezone.utc)).total_seconds()

    act_on_model()
    time.sleep(sleep_seconds)
