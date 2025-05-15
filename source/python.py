import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from itertools import combinations
from collections import defaultdict

# === Load Data ===
folder_path = "/Users/francopapalardo-aleo/Desktop/repos/TradingAI/data/"
column_names = ['datetime', 'open', 'high', 'low', 'close', 'volume']
df_list = []

for filename in os.listdir(folder_path):
    if filename.endswith(('.csv', '.txt')):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path, sep=';', header=None, names=column_names)
        df['source_file'] = filename
        df_list.append(df)

df = pd.concat(df_list, ignore_index=True)
df['datetime'] = pd.to_datetime(df['datetime'])
# Resample to 5-minute candles
df = df.sort_values('datetime').reset_index(drop=True)
df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

# === Feature Engineering ===
df['ema_9'] = ta.ema(df['close'], length=9)
df['ema_21'] = ta.ema(df['close'], length=21)
df['rsi'] = ta.rsi(df['close'], length=14)
df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
macd = ta.macd(df['close'])
df['macd'] = macd['MACDh_12_26_9']
df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
# df['vwap_diff'] = df['close'] - df['vwap']

def choppiness_index(high, low, close, length=14):
    tr = ta.true_range(high=high, low=low, close=close)
    atr_sum = tr.rolling(length).sum()
    high_max = high.rolling(length).max()
    low_min = low.rolling(length).min()
    return 100 * np.log10(atr_sum / (high_max - low_min)) / np.log10(length)

# === Add Feature ===
df['chop_index'] = choppiness_index(df['high'], df['low'], df['close'])

# === Strategy Setup ===
df = df.dropna().reset_index(drop=True)
TICK_VALUE = 5
SL_ATR_MULT = 1.5
TP_ATR_MULT = 3.0
TRAIL_START_MULT = 2.5
TRAIL_STOP_MULT = 1.0
MAX_CONTRACTS = 5
thresholds = [0.001, 0.002]
confidence_levels = [0.4, 0.5]

features = [
    'rsi', 'macd', 'ema_9', 'ema_21', 'volume', 'chop_index',
    'atr_14'
    # , 'vwap_diff'
]

feature_funcs = {
    'atr_14': lambda r: r['atr_14'] > 10,
    # 'vwap_diff': lambda r: r['vwap_diff'] < -40,
    'chop_index': lambda r: r['chop_index'] < 60,
    'rsi': lambda r: r['rsi'] < 50,
    'ema_cross': lambda r: r['ema_9'] > r['ema_21'],
    'macd': lambda r: r['macd'] > 0
}

def is_same_session(start_time, end_time):
    session_start = start_time.replace(hour=18, minute=0, second=0)
    if start_time.hour < 18:
        session_start -= timedelta(days=1)
    session_end = session_start + timedelta(hours=23)
    return session_start <= start_time <= session_end and session_start <= end_time <= session_end

combo_trades = defaultdict(set)

def combo_overlap(c1, c2):
    a, b = combo_trades[frozenset(c1)], combo_trades[frozenset(c2)]
    if not a or not b:
        return 1.0
    return len(a & b) / min(len(a), len(b))

# === Train & Evaluate ===
combo_stats = []
for CONF_THRESHOLD in confidence_levels:
    print(f"Running strategy for {CONF_THRESHOLD}")
    for threshold in thresholds:
        df['future_close'] = df['close'].shift(-1)
        move = df['future_close'] - df['close']
        df['target'] = 1
        df.loc[move < -df['close'] * threshold, 'target'] = -1
        df.loc[(move >= -df['close'] * threshold) & (move <= df['close'] * threshold), 'target'] = 0

        labeled = df.dropna(subset=['target']).copy()
        X = labeled[features]
        y = labeled['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)

        for size in range(2, len(features)):
            for combo in combinations(feature_funcs.items(), size):
                name_combo = [k for k, _ in combo]
                func_combo = [f for _, f in combo]
                print(f"Running combination {name_combo}")
                temp_trades_data = []

                for idx in X_test.index:
                    row = labeled.loc[idx]
                    score = sum(f(row) for f in func_combo)
                    confidence = score / size
                    if confidence < CONF_THRESHOLD or row['target'] == 0 or idx >= len(df) - 6:
                        continue

                    entry_price = row['close']
                    entry_time = row['datetime']
                    side = 'long' if row['target'] == 1 else 'short'
                    atr = row['atr_14']
                    sl_price = entry_price - SL_ATR_MULT * atr if side == 'long' else entry_price + SL_ATR_MULT * atr
                    tp_price = entry_price + TP_ATR_MULT * atr if side == 'long' else entry_price - TP_ATR_MULT * atr
                    trail_trigger = entry_price + TRAIL_START_MULT * atr if side == 'long' else entry_price - TRAIL_START_MULT * atr
                    trail_stop = None

                    max_price, min_price = entry_price, entry_price
                    exit_price, exit_time = None, None

                    for fwd in range(1, 6):
                        fwd_idx = idx + fwd
                        if fwd_idx >= len(df): break
                        fwd_row = df.loc[fwd_idx]
                        max_price = max(max_price, fwd_row['high'])
                        min_price = min(min_price, fwd_row['low'])

                        # SL hit
                        if (side == 'long' and fwd_row['low'] <= sl_price) or (side == 'short' and fwd_row['high'] >= sl_price):
                            exit_price = sl_price
                            exit_time = fwd_row['datetime']
                            break

                        # TP hit
                        if (side == 'long' and fwd_row['high'] >= tp_price) or (side == 'short' and fwd_row['low'] <= tp_price):
                            exit_price = tp_price
                            exit_time = fwd_row['datetime']
                            break

                        # Breakeven/trailing
                        if side == 'long' and fwd_row['high'] >= trail_trigger:
                            trail_stop = fwd_row['close'] - TRAIL_STOP_MULT * atr
                        if side == 'short' and fwd_row['low'] <= trail_trigger:
                            trail_stop = fwd_row['close'] + TRAIL_STOP_MULT * atr

                        if trail_stop:
                            if (side == 'long' and fwd_row['low'] <= trail_stop) or (side == 'short' and fwd_row['high'] >= trail_stop):
                                exit_price = trail_stop
                                exit_time = fwd_row['datetime']
                                break

                    if exit_price is None:
                        exit_price = df.loc[idx + 5, 'close']
                        exit_time = df.loc[idx + 5, 'datetime']

                    if not is_same_session(entry_time, exit_time): continue

                    mfe = max_price - entry_price if side == 'long' else entry_price - min_price
                    mae = entry_price - min_price if side == 'long' else max_price - entry_price
                    rr_ratio = mfe / (mae + 1e-9)
                    if rr_ratio < 1.2: continue

                    pnl = (exit_price - entry_price) * TICK_VALUE if side == 'long' else (entry_price - exit_price) * TICK_VALUE
                    temp_trades_data.append({'pnl': pnl, 'mfe': mfe, 'mae': mae})

                results = pd.DataFrame(temp_trades_data)
                pnl_total = results['pnl'].sum() if not results.empty else 0
                trades = len(results)
                win_rate = (results['pnl'] > 0).mean() if not results.empty else 0
                avg_mfe = results['mfe'].mean() if not results.empty else 0
                avg_mae = results['mae'].mean() if not results.empty else 0
                max_win = results['pnl'].max() if not results.empty else 0
                max_loss = results['pnl'].min() if not results.empty else 0

                combo_stats.append({
                    'features': name_combo,
                    'pnl': pnl_total,
                    'trades': trades,
                    'win_rate': win_rate,
                    'avg_mfe': avg_mfe,
                    'avg_mae': avg_mae,
                    'max_win': max_win,
                    'max_loss': max_loss,
                    'threshold': threshold,
                    'confidence': CONF_THRESHOLD,
                    'results': results
                })

# === Sort & Plot Top 2 Combos ===
filtered = [s for s in combo_stats if s['win_rate'] >= 0.2]
sorted_combos = sorted(
    filtered,
    key=lambda x: (
        x['pnl'],
        x['trades'],
        x['win_rate'],
        x['avg_mfe'],
        -x['avg_mae'],
        -x['max_loss']
    ),
    reverse=True
)
top_2 = sorted_combos[:2]
best_combo = sorted_combos[0]
best_threshold = best_combo['threshold']
best_confidence = best_combo['confidence']
print(f"\n🏆 Best Overall Threshold: {best_threshold}")
print(f"📈 Best Overall Confidence: {best_confidence}")

for i, combo in enumerate(top_2):
    print(f"\n🔥 Top Combo #{i+1} | Confidence={combo['confidence']}: {combo['features']} | PnL=${combo['pnl']:.2f}, Trades={combo['trades']}, WinRate={combo['win_rate']:.2%}, MFE={combo['avg_mfe']:.2f}, MAE={combo['avg_mae']:.2f}, MaxWin=${combo['max_win']:.2f}, MaxLoss=${combo['max_loss']:.2f}")
    equity_curve = combo['results']['pnl'].cumsum()
    plt.figure(figsize=(10, 4))
    plt.plot(equity_curve, label=f"Combo #{i+1}")
    plt.title(f"Equity Curve - Combo #{i+1}")
    plt.xlabel("Trades")
    plt.ylabel("Cumulative PnL ($)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

print("\n📊 Combo Overlap Matrix (Top 5 Combos):")
for i in range(min(5, len(sorted_combos))):
    for j in range(i + 1, min(5, len(sorted_combos))):
        a = sorted_combos[i]['features']
        b = sorted_combos[j]['features']
        overlap = combo_overlap(a, b)
        print(f"{a} vs {b} → Overlap: {overlap:.2%}")

print("\n📊 Combo Overlap Matrix (Top 5 Combos):")
for i in range(min(5, len(sorted_combos))):
    for j in range(i + 1, min(5, len(sorted_combos))):
        a = sorted_combos[i]['features']
        b = sorted_combos[j]['features']
        overlap = combo_overlap(a, b)
        print(f"{a} vs {b} → Overlap: {overlap:.2%}")

# === Print Summary Table ===
print("\n📋 Feature Combo Summary:")
for stat in sorted_combos[:10]:
    print(f"{stat['features']} | Conf={stat['confidence']} | PnL=${stat['pnl']:.2f}, Trades={stat['trades']}, WinRate={stat['win_rate']:.2%}, MFE={stat['avg_mfe']:.2f}, MAE={stat['avg_mae']:.2f}, MaxWin=${stat['max_win']:.2f}, MaxLoss=${stat['max_loss']:.2f}")