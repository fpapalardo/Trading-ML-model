import numpy as np
import pandas as pd
from collections import defaultdict

def evaluate_regression(
    X_test, preds_stack, labeled, df,
    avoid_funcs,
    SL_ATR_MULT, TP_ATR_MULT, TRAIL_START_MULT, TRAIL_STOP_MULT, TICK_VALUE,
    is_same_session,
    long_thresh,
    short_thresh,
    base_contracts=1,
    max_contracts=5,
    skip_weak_conf=False,
    weak_conf_zscore=0.2,
    stack_weight=0.4,
    cnn_weight=0.3
):
    temp_trades_data = []
    skipped_trades = 0
    avoid_hits = defaultdict(int)
    long_trades = 0
    short_trades = 0

    X_test_idx = X_test.index.to_list()
    preds_array = preds_stack

    # === Confidence & Position Size ===
    zscores = (preds_array - preds_array.mean()) / (preds_array.std() + 1e-9)
    zscores = np.clip(zscores, -3.0, 3.0)
    conf_scores = np.clip(np.abs(zscores), 0, 2.0)
    position_sizes = base_contracts + (max_contracts - base_contracts) * (conf_scores / 2.0)
    position_sizes = np.round(position_sizes, 0)

    i = 0
    while i < len(X_test_idx):
        idx = X_test_idx[i]

        if df.index.get_loc(idx) + 1 >= len(df):
            skipped_trades += 1
            i += 1
            continue

        row = labeled.loc[idx]
        vol_adj_pred = preds_array[i]
        conf = conf_scores[i]
        size = position_sizes[i]

        if vol_adj_pred >= long_thresh:
            side = 'long'
            long_trades += 1
        elif vol_adj_pred <= short_thresh:
            side = 'short'
            short_trades += 1
        else:
            skipped_trades += 1
            i += 1
            continue

        skip_trade = False
        for name, f in avoid_funcs.items():
            try:
                if f(row):
                    avoid_hits[name] += 1
                    skip_trade = True
            except:
                continue
        if skip_trade:
            skipped_trades += 1
            i += 1
            continue

        # --- Trade Simulation ---
        entry_price = row['open']
        entry_time = row.name
        atr = row['ATR_14_5m']

        expected_move = abs(vol_adj_pred) * entry_price
        min_tp = 0.005 * entry_price
        max_tp = TP_ATR_MULT * atr
        tp_move = np.clip(expected_move, min_tp, max_tp)
        tp_price = entry_price + tp_move if side == 'long' else entry_price - tp_move

        sl_move = SL_ATR_MULT * atr
        if sl_move > tp_move:
            sl_move = tp_move

        sl_price = entry_price - sl_move if side == 'long' else entry_price + sl_move

        trail_trigger = entry_price + TRAIL_START_MULT * atr if side == 'long' else entry_price - TRAIL_START_MULT * atr
        trail_stop = None
        max_price, min_price = entry_price, entry_price
        exit_price, exit_time = None, None

        fwd_idx = labeled.index.get_loc(idx) + 1
        while fwd_idx < len(df):
            fwd_row = labeled.iloc[fwd_idx]
            max_price = max(max_price, fwd_row['high'])
            min_price = min(min_price, fwd_row['low'])

            if (side == 'long' and fwd_row['low'] <= sl_price) or (side == 'short' and fwd_row['high'] >= sl_price):
                exit_price = sl_price
                exit_time = fwd_row.name
                break

            if (side == 'long' and fwd_row['high'] >= tp_price) or (side == 'short' and fwd_row['low'] <= tp_price):
                exit_price = tp_price
                exit_time = fwd_row.name
                break

            if side == 'long' and fwd_row['high'] >= trail_trigger:
                trail_stop = fwd_row['close'] - TRAIL_STOP_MULT * atr
            if side == 'short' and fwd_row['low'] <= trail_trigger:
                trail_stop = fwd_row['close'] + TRAIL_STOP_MULT * atr

            if trail_stop:
                if (side == 'long' and fwd_row['low'] <= trail_stop) or (side == 'short' and fwd_row['high'] >= trail_stop):
                    exit_price = trail_stop
                    exit_time = fwd_row.name
                    break

            fwd_idx += 1

        if exit_price is None:
            fallback_row = labeled.iloc[-1]
            exit_price = fallback_row['close']
            exit_time = fallback_row.name

        if not is_same_session(entry_time, exit_time):
            i += 1
            continue

        GROSS_PNL = (exit_price - entry_price) * TICK_VALUE * size if side == 'long' else (entry_price - exit_price) * TICK_VALUE * size
        COMMISSION = 3.98 * size
        pnl = GROSS_PNL - COMMISSION
        mfe = max_price - entry_price if side == 'long' else entry_price - min_price
        mae = entry_price - min_price if side == 'long' else max_price - entry_price

        temp_trades_data.append({
            'entry_time': entry_time,
            'exit_time': exit_time,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'mfe': mfe,
            'mae': mae,
            'gross_pnl': GROSS_PNL,
            'vol_adj_pred': vol_adj_pred,
            'confidence': conf,
            'position_size': size,
        })

        while i < len(X_test_idx) and labeled.loc[X_test_idx[i]].name <= exit_time:
            i += 1

    # === Metrics ===
    results = pd.DataFrame(temp_trades_data)
    pnl_total = results['pnl'].sum() if not results.empty else 0
    trades = len(results)
    win_rate = (results['pnl'] > 0).mean() if not results.empty else 0
    expectancy = results['pnl'].mean() if not results.empty else 0
    profit_factor = results[results['pnl'] > 0]['pnl'].sum() / abs(results[results['pnl'] < 0]['pnl'].sum()) if not results.empty and (results['pnl'] < 0).any() else np.nan
    sharpe = results['pnl'].mean() / (results['pnl'].std() + 1e-9) * np.sqrt(trades) if trades > 1 else 0

    # === Average Confidence for Wins and Losses ===
    avg_confidence_win = abs(results[results['pnl'] > 0]['vol_adj_pred']).mean() if not results.empty else np.nan
    avg_confidence_loss = abs(results[results['pnl'] <= 0]['vol_adj_pred']).mean() if not results.empty else np.nan

    return {
        'pnl': pnl_total,
        'trades': trades,
        'win_rate': win_rate,
        'expectancy': expectancy,
        'profit_factor': profit_factor,
        'sharpe': sharpe,
        'long_trades': long_trades,
        'short_trades': short_trades,
        'avoid_hits': dict(avoid_hits),
        'threshold': long_thresh,
        'results': results,
        'avg_confidence_win': avg_confidence_win,
        'avg_confidence_loss': avg_confidence_loss
    }
