import numpy as np
import pandas as pd
from collections import defaultdict
from numba import njit

from collections import defaultdict
import numpy as np
import pandas as pd
from datetime import timedelta

def session_key(ts: pd.Timestamp) -> pd.Timestamp:
    # shift back 18 h, then floor to midnight to get a unique session “date”
    return (ts - timedelta(hours=18)).normalize()

def is_same_session(start_time: pd.Timestamp, end_time: pd.Timestamp) -> bool:
    return session_key(start_time) == session_key(end_time)

def evaluate_regression(
    X_test, 
    preds_stack, 
    labeled, 
    df,
    avoid_funcs,
    TRAIL_START_MULT, 
    TRAIL_STOP_MULT,
    #–––  You used to pass a single SL and TP multiplier.  Now we pass three of each:
    TICK_VALUE,
    long_thresh,
    short_thresh,
    is_same_session=is_same_session,
    base_contracts=1,
    max_contracts=5, 
    SL_ATR_MULT_TREND: float = 1.0, SL_ATR_MULT_CHOP: float = 0.5, SL_ATR_MULT_DEFAULT: float  = 1.0,
    TP_ATR_MULT_TREND: float = 3.0, TP_ATR_MULT_CHOP: float = 1.0, TP_ATR_MULT_DEFAULT: float = 2.0
):
    """
    Backtest a set of “vol‐adjusted” predictions (preds_stack) on the `labeled` DataFrame.
    At each bar N we decide to go long/short on bar N+1 (using same‐session rules, avoid_funcs, etc.).
    We now inject regime‐aware SL/TP by reading row['ADX_14_5min'] and row['CHOP_14_1_100_5min'].

    -----------------------------------------------------------------------
    PARAMETERS (only noting the regime‐related additions):

      SL_ATR_MULT_TREND      – when ADX>25 and CHOP<50  (strong trend), use this ATR for stop loss
      TP_ATR_MULT_TREND      – when ADX>25 and CHOP<50  (strong trend), use this ATR for take profit

      SL_ATR_MULT_CHOP       – when CHOP>60 (choppy market), use this ATR for stop loss
      TP_ATR_MULT_CHOP       – when CHOP>60 (choppy market), use this ATR for take profit

      SL_ATR_MULT_DEFAULT    – otherwise (neither strongly trending nor choppy), use this ATR for SL
      TP_ATR_MULT_DEFAULT    – otherwise (neither strongly trending nor choppy), use this ATR for TP

    RETURNS:
      A dict containing PnL, trade statistics, and a DataFrame 'results' where each row includes:
        - entry_time, exit_time, side, entry_price, exit_price, pnl, mfe, mae, gross_pnl
        - vol_adj_pred, confidence, position_size, exit_reason, used_trailing
    """
    from collections import defaultdict

    temp_trades_data = []
    skipped_trades = 0
    avoid_hits = defaultdict(int)
    long_trades = 0
    short_trades = 0

    X_test_idx   = X_test.index.to_list()
    preds_array  = preds_stack

    # Convert raw preds_stack into a “confidence score” (z‐score clipped to ±3)
    zscores        = (preds_array - preds_array.mean()) / (preds_array.std() + 1e-9)
    zscores        = np.clip(zscores, -3.0, 3.0)
    conf_scores    = np.clip(np.abs(zscores), 0, 2.0)
    position_sizes = base_contracts + (max_contracts - base_contracts) * (conf_scores / 2.0)
    position_sizes = np.round(position_sizes, 0)

    i = 0
    while i < len(X_test_idx):
        idx      = X_test_idx[i]
        idx_loc  = labeled.index.get_loc(idx)
        
        # If we cannot enter on N+1 (beyond data), skip
        if idx_loc + 1 >= len(labeled):
            skipped_trades += 1
            i += 1
            continue

        row       = labeled.iloc[idx_loc]             # bar N (features + actual reg targets)
        entry_row = labeled.iloc[idx_loc + 1]         # bar N+1 (where we actually enter)

        vol_adj_pred = preds_array[i]
        conf         = conf_scores[i]
        size         = position_sizes[i]

        # Decide direction
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

        # Apply “avoid” filters (e.g. avoid on news, avoid on Bollinger squeeze, etc.)
        skip_trade = False
        for name, f in avoid_funcs.items():
            try:
                if f(row):  
                    avoid_hits[name] += 1
                    skip_trade = True
            except:
                pass
        if skip_trade:
            skipped_trades += 1
            i += 1
            continue

        entry_price = entry_row['open']
        entry_time  = row.name - pd.Timedelta(minutes=5)  
        atr         = row['ATR_14_5min']
        adx_val     = row['ADX_14_5min']
        chop_val    = row['CHOP_14_1_100_5min']

        # ──── REGIME LOGIC: choose SL/TP‐ATR multipliers based on (ADX, CHOP) ────
        if (adx_val > 25.0) and (chop_val < 50.0):
            # strong trend
            sl_mult = SL_ATR_MULT_TREND
            tp_mult = TP_ATR_MULT_TREND
        elif (chop_val > 60.0):
            # choppy
            sl_mult = SL_ATR_MULT_CHOP
            tp_mult = TP_ATR_MULT_CHOP
        else:
            # default
            sl_mult = SL_ATR_MULT_DEFAULT
            tp_mult = TP_ATR_MULT_DEFAULT
        # ────────────────────────────────────────────────────────────────────

        # If ATR or entry price is nonsense, skip
        if entry_price <= 0 or atr <= 0:
            skipped_trades += 1
            i += 1
            continue

        # ──── Compute actual SL/TP distances in price‐units ────
        sl_move = sl_mult * atr
        tp_move = tp_mult * atr

        # enforce a minimum TP of 0.5% of entry_price
        min_tp = 0.005 * entry_price
        tp_move = np.clip(tp_move, min_tp, tp_move)

        # if your SL distance is bigger than TP, shrink SL to TP
        if sl_move > tp_move:
            sl_move = tp_move

        tp_price = entry_price + tp_move if side == 'long' else entry_price - tp_move
        sl_price = entry_price - sl_move  if side == 'long' else entry_price + sl_move

        # ──── TWO‐PHASE TRAILING STOP SETUP ────
        trailing_activated = False
        trail_stop         = None

        # “activation price” = entry_price ± (TRAIL_START_MULT * ATR)
        # later, once activated, we’ll ratchet behind the new highs/lows
        if TRAIL_START_MULT > 0 and TRAIL_STOP_MULT > 0:
            if side == 'long':
                trail_trigger_price = entry_price + TRAIL_START_MULT * atr
            else:  # side == 'short'
                trail_trigger_price = entry_price - TRAIL_START_MULT * atr
        else:
            trail_trigger_price = None  # means “never activate” if either multiplier is zero

        max_price = entry_price
        min_price = entry_price
        exit_price = None
        exit_time  = None

        fwd_idx = labeled.index.get_loc(entry_row.name) + 1
        while fwd_idx < len(df):
            fwd_row = labeled.iloc[fwd_idx]
            high_j  = fwd_row["high"]
            low_j   = fwd_row["low"]
            close_j = fwd_row["close"]

            # track the highest/lowest price seen so far during this trade
            if high_j > max_price:
                max_price = high_j
            if low_j < min_price:
                min_price = low_j

            # ──── 1) Check SL first ────
            if side == 'long' and low_j <= sl_price:
                exit_price  = sl_price
                exit_time   = fwd_row.name
                exit_reason = 'SL'
                used_trailing = False
                break
            if side == 'short' and high_j >= sl_price:
                exit_price  = sl_price
                exit_time   = fwd_row.name
                exit_reason = 'SL'
                used_trailing = False
                break

            # ──── 2) Check TP second ────
            if side == 'long' and high_j >= tp_price:
                exit_price  = tp_price
                exit_time   = fwd_row.name
                exit_reason = 'TP'
                used_trailing = False
                break
            if side == 'short' and low_j <= tp_price:
                exit_price  = tp_price
                exit_time   = fwd_row.name
                exit_reason = 'TP'
                used_trailing = False
                break

            # ──── 3) TRAILING STOP LOGIC ────
            if trail_trigger_price is not None:
                if not trailing_activated:
                    # not yet activated—wait for price to cross trail_trigger_price
                    if side == 'long' and high_j >= trail_trigger_price:
                        # first activation: move stop all the way up to breakeven
                        trail_stop = entry_price
                        trailing_activated = True
                    elif side == 'short' and low_j <= trail_trigger_price:
                        trail_stop = entry_price
                        trailing_activated = True
                else:
                    # already activated—ratchet trailing stop behind the new high/low
                    if side == 'long':
                        new_trail = max_price - TRAIL_STOP_MULT * atr
                        # do not move the stop below breakeven; only ratchet up
                        if new_trail > trail_stop:
                            trail_stop = new_trail
                    else:  # short
                        new_trail = min_price + TRAIL_STOP_MULT * atr
                        if new_trail < trail_stop:
                            trail_stop = new_trail

                # now, if trailing_activated and price has hit the trail_stop, exit
                if trailing_activated:
                    if side == 'long' and low_j <= trail_stop:
                        exit_price  = trail_stop
                        exit_time   = fwd_row.name
                        exit_reason = 'TRAIL'
                        used_trailing = True
                        break
                    if side == 'short' and high_j >= trail_stop:
                        exit_price  = trail_stop
                        exit_time   = fwd_row.name
                        exit_reason = 'TRAIL'
                        used_trailing = True
                        break

            # ──── 4) SESSION END ────
            # If price never hit SL, TP, or TRAIL, but now we left the same session,
            # force‐exit at this bar’s close:
            if not is_same_session(entry_time, fwd_row.name):
                exit_price    = close_j
                exit_time     = fwd_row.name
                exit_reason   = 'SESSION_END'
                used_trailing = trailing_activated  # could be True or False
                break

            fwd_idx += 1

        # ──── 5) If we never hit any exit above, force‐exit at the very last bar ────
        if exit_price is None:
            fallback_row    = labeled.iloc[-1]
            exit_price      = fallback_row['close']
            exit_time       = fallback_row.name
            exit_reason     = 'FORCED_END'
            used_trailing   = trailing_activated

        # ──── 6) Compute PnL, MFE, MAE exactly as before ────
        GROSS_PNL  = (
            (exit_price - entry_price) * TICK_VALUE * size
            if side == 'long'
            else (entry_price - exit_price) * TICK_VALUE * size
        )
        COMMISSION = 3.98 * size
        pnl        = GROSS_PNL - COMMISSION

        mfe = (max_price - entry_price) * TICK_VALUE if side == 'long' else (entry_price - min_price) * TICK_VALUE
        mae = (entry_price - min_price) * TICK_VALUE if side == 'long' else (max_price - entry_price) * TICK_VALUE

        temp_trades_data.append({
            'entry_time':     entry_time,
            'exit_time':      exit_time,
            'side':           side,
            'entry_price':    entry_price,
            'exit_price':     exit_price,
            'pnl':            pnl,
            'mfe':            mfe,
            'mae':            mae,
            'gross_pnl':      GROSS_PNL,
            'vol_adj_pred':   vol_adj_pred,
            'confidence':     conf,
            'position_size':  size,
            'exit_reason':    exit_reason,
            'used_trailing':  used_trailing,
        })

        # ──── 7) Advance `i` past this trade’s exit time ────
        while i < len(X_test_idx) and labeled.loc[X_test_idx[i]].name <= exit_time:
            i += 1

    # Compile stats
    results     = pd.DataFrame(temp_trades_data)
    pnl_total   = results['pnl'].sum() if not results.empty else 0
    trades      = len(results)
    win_rate    = (results['pnl'] > 0).mean() if not results.empty else 0
    expectancy  = results['pnl'].mean() if not results.empty else 0
    profit_factor = (
        results[results['pnl'] > 0]['pnl'].sum()
        / abs(results[results['pnl'] < 0]['pnl'].sum())
        if (not results.empty) and (results['pnl'] < 0).any()
        else np.nan
    )
    sharpe = (
        (results['pnl'].mean() / (results['pnl'].std() + 1e-9)) * np.sqrt(trades)
        if trades > 1 else 0
    )
    avg_confidence_win  = abs(results[results['pnl'] > 0]['vol_adj_pred']).mean() if not results.empty else np.nan
    avg_confidence_loss = abs(results[results['pnl'] <= 0]['vol_adj_pred']).mean() if not results.empty else np.nan

    return {
        'pnl':                  pnl_total,
        'trades':               trades,
        'win_rate':             win_rate,
        'expectancy':           expectancy,
        'profit_factor':        profit_factor,
        'sharpe':               sharpe,
        'long_trades':          long_trades,
        'short_trades':         short_trades,
        'avoid_hits':           dict(avoid_hits),
        'threshold':            long_thresh,
        'results':              results,
        'avg_confidence_win':   avg_confidence_win,
        'avg_confidence_loss':  avg_confidence_loss
    }

# def evaluate_classification(
#     X_test, preds_stack, labeled,
#     avoid_funcs,
#     TRAIL_START_MULT, TRAIL_STOP_MULT, TICK_VALUE,
#     SL_ATR_MULT=1.5, TP_ATR_MULT=2.0,
#     is_same_session=is_same_session,
#     base_contracts=1,
#     max_contracts=1,
# ):
#     temp_trades_data = []
#     skipped_trades = 0
#     avoid_hits = defaultdict(int)
#     long_trades = 0
#     short_trades = 0

#     X_test_idx = X_test.index.to_list()

#     i = 0
#     while i < len(X_test_idx):
#         idx      = X_test_idx[i]
#         idx_loc  = labeled.index.get_loc(idx)
        
#         # If we cannot enter on N+1 (beyond data), skip
#         if idx_loc + 1 >= len(labeled):
#             skipped_trades += 1
#             i += 1
#             continue

#         row       = labeled.iloc[idx_loc]             # bar N (features + actual reg targets)
#         entry_row = labeled.iloc[idx_loc + 1]         # bar N+1 (where we actually enter)

#         pred        = preds_stack[i]
#         size         = base_contracts

#         # Decide direction
#         if pred == 1:
#             side = 'long'
#             long_trades += 1
#         elif pred == 2:
#             side = 'short'
#             short_trades += 1
#         else:
#             skipped_trades += 1
#             i += 1
#             continue

#         # Apply “avoid” filters (e.g. avoid on news, avoid on Bollinger squeeze, etc.)
#         skip_trade = False
#         for name, f in avoid_funcs.items():
#             try:
#                 if f(row):  
#                     avoid_hits[name] += 1
#                     skip_trade = True
#             except:
#                 pass
#         if skip_trade:
#             skipped_trades += 1
#             i += 1
#             continue

#         entry_price = entry_row['open']
#         entry_time  = row.name - pd.Timedelta(minutes=5)  
#         atr         = row['ATR_14_5min']

#         sl_mult = SL_ATR_MULT
#         tp_mult = TP_ATR_MULT
#         # ────────────────────────────────────────────────────────────────────

#         # If ATR or entry price is nonsense, skip
#         if entry_price <= 0 or atr <= 0:
#             skipped_trades += 1
#             i += 1
#             continue

#         # ──── Compute actual SL/TP distances in price‐units ────
#         sl_move = sl_mult * atr
#         tp_move = tp_mult * atr

#         tp_price = entry_price + tp_move if side == 'long' else entry_price - tp_move
#         sl_price = entry_price - sl_move  if side == 'long' else entry_price + sl_move

#         # ──── TWO‐PHASE TRAILING STOP SETUP ────
#         trailing_activated = False
#         trail_stop         = None

#         # “activation price” = entry_price ± (TRAIL_START_MULT * ATR)
#         # later, once activated, we’ll ratchet behind the new highs/lows
#         if TRAIL_START_MULT > 0 and TRAIL_STOP_MULT > 0:
#             if side == 'long':
#                 trail_trigger_price = entry_price + TRAIL_START_MULT * atr
#             else:  # side == 'short'
#                 trail_trigger_price = entry_price - TRAIL_START_MULT * atr
#         else:
#             trail_trigger_price = None  # means “never activate” if either multiplier is zero

#         max_price = entry_price
#         min_price = entry_price
#         exit_price = None
#         exit_time  = None

#         fwd_idx = labeled.index.get_loc(entry_row.name) + 1
#         while fwd_idx < len(labeled):
#             fwd_row = labeled.iloc[fwd_idx]
#             high_j  = fwd_row["high"]
#             low_j   = fwd_row["low"]
#             close_j = fwd_row["close"]

#             # track the highest/lowest price seen so far during this trade
#             if high_j > max_price:
#                 max_price = high_j
#             if low_j < min_price:
#                 min_price = low_j

#             # ──── 1) Check SL first ────
#             if side == 'long' and low_j <= sl_price:
#                 exit_price  = sl_price
#                 exit_time   = fwd_row.name
#                 exit_reason = 'SL'
#                 used_trailing = False
#                 break
#             if side == 'short' and high_j >= sl_price:
#                 exit_price  = sl_price
#                 exit_time   = fwd_row.name
#                 exit_reason = 'SL'
#                 used_trailing = False
#                 break

#             # ──── 2) Check TP second ────
#             if side == 'long' and high_j >= tp_price:
#                 exit_price  = tp_price
#                 exit_time   = fwd_row.name
#                 exit_reason = 'TP'
#                 used_trailing = False
#                 break
#             if side == 'short' and low_j <= tp_price:
#                 exit_price  = tp_price
#                 exit_time   = fwd_row.name
#                 exit_reason = 'TP'
#                 used_trailing = False
#                 break

#             # ──── 3) TRAILING STOP LOGIC ────
#             if trail_trigger_price is not None:
#                 if not trailing_activated:
#                     # not yet activated—wait for price to cross trail_trigger_price
#                     if side == 'long' and high_j >= trail_trigger_price:
#                         # first activation: move stop all the way up to breakeven
#                         trail_stop = entry_price
#                         trailing_activated = True
#                     elif side == 'short' and low_j <= trail_trigger_price:
#                         trail_stop = entry_price
#                         trailing_activated = True
#                 else:
#                     # already activated—ratchet trailing stop behind the new high/low
#                     if side == 'long':
#                         new_trail = max_price - TRAIL_STOP_MULT * atr
#                         # do not move the stop below breakeven; only ratchet up
#                         if new_trail > trail_stop:
#                             trail_stop = new_trail
#                     else:  # short
#                         new_trail = min_price + TRAIL_STOP_MULT * atr
#                         if new_trail < trail_stop:
#                             trail_stop = new_trail

#                 # now, if trailing_activated and price has hit the trail_stop, exit
#                 if trailing_activated:
#                     if side == 'long' and low_j <= trail_stop:
#                         exit_price  = trail_stop
#                         exit_time   = fwd_row.name
#                         exit_reason = 'TRAIL'
#                         used_trailing = True
#                         break
#                     if side == 'short' and high_j >= trail_stop:
#                         exit_price  = trail_stop
#                         exit_time   = fwd_row.name
#                         exit_reason = 'TRAIL'
#                         used_trailing = True
#                         break

#             # ──── 4) SESSION END ────
#             # If price never hit SL, TP, or TRAIL, but now we left the same session,
#             # force‐exit at this bar’s close:
#             if not is_same_session(entry_time, fwd_row.name):
#                 exit_price    = close_j
#                 exit_time     = fwd_row.name
#                 exit_reason   = 'SESSION_END'
#                 used_trailing = trailing_activated  # could be True or False
#                 break

#             fwd_idx += 1

#         # ──── 5) If we never hit any exit above, force‐exit at the very last bar ────
#         if exit_price is None:
#             fallback_row    = labeled.iloc[-1]
#             exit_price      = fallback_row['close']
#             exit_time       = fallback_row.name
#             exit_reason     = 'FORCED_END'
#             used_trailing   = trailing_activated

#         # ──── 6) Compute PnL, MFE, MAE exactly as before ────
#         GROSS_PNL  = (
#             (exit_price - entry_price) * TICK_VALUE * size
#             if side == 'long'
#             else (entry_price - exit_price) * TICK_VALUE * size
#         )
#         COMMISSION = 3.98 * size
#         pnl        = GROSS_PNL - COMMISSION

#         mfe = (max_price - entry_price) * TICK_VALUE if side == 'long' else (entry_price - min_price) * TICK_VALUE
#         mae = (entry_price - min_price) * TICK_VALUE if side == 'long' else (max_price - entry_price) * TICK_VALUE

#         temp_trades_data.append({
#             'entry_time':     entry_time,
#             'exit_time':      exit_time,
#             'side':           side,
#             'entry_price':    entry_price,
#             'exit_price':     exit_price,
#             'pnl':            pnl,
#             'mfe':            mfe,
#             'mae':            mae,
#             'gross_pnl':      GROSS_PNL,
#             'position_size':  size,
#             'exit_reason':    exit_reason,
#             'used_trailing':  used_trailing,
#         })

#         # ──── 7) Advance `i` past this trade’s exit time ────
#         while i < len(X_test_idx) and labeled.loc[X_test_idx[i]].name <= exit_time:
#             i += 1

#     # Compile stats
#     results     = pd.DataFrame(temp_trades_data)
#     pnl_total   = results['pnl'].sum() if not results.empty else 0
#     trades      = len(results)
#     win_rate    = (results['pnl'] > 0).mean() if not results.empty else 0
#     expectancy  = results['pnl'].mean() if not results.empty else 0
#     profit_factor = (
#         results[results['pnl'] > 0]['pnl'].sum()
#         / abs(results[results['pnl'] < 0]['pnl'].sum())
#         if (not results.empty) and (results['pnl'] < 0).any()
#         else np.nan
#     )
#     sharpe = (
#         (results['pnl'].mean() / (results['pnl'].std() + 1e-9)) * np.sqrt(trades)
#         if trades > 1 else 0
#     )

#     return {
#         'pnl':                  pnl_total,
#         'trades':               trades,
#         'win_rate':             win_rate,
#         'expectancy':           expectancy,
#         'profit_factor':        profit_factor,
#         'sharpe':               sharpe,
#         'long_trades':          long_trades,
#         'short_trades':         short_trades,
#         'avoid_hits':           dict(avoid_hits),
#         'results':              results,
#     }

def calculate_max_drawdown(balance_history_df):
    """
    Calculates the maximum drawdown from a balance history DataFrame.
    Used by the crypto evaluation function.
    """
    if balance_history_df.empty or 'balance' not in balance_history_df.columns:
        return 0.0
    
    balance_series = balance_history_df['balance']
    cumulative_max = balance_series.cummax()
    drawdown = (cumulative_max - balance_series) / cumulative_max
    max_drawdown = drawdown.max()
    return max_drawdown

@njit
def simulate_trade_crypto(
    start_i,
    session_last_i,
    side_code,
    entry_price,
    atr,
    sl_mult,
    tp_mult,
    trail_start_mult,
    trail_stop_mult,
    opens,
    highs,
    lows,
    closes
):
    """Original simulate_trade function - unchanged"""
    max_p = entry_price
    min_p = entry_price

    # only enable trailing when both multipliers > 0
    use_trailing = (trail_start_mult > 0.0) and (trail_stop_mult > 0.0)
    if side_code == 1:
        trail_trigger = entry_price + trail_start_mult * atr if use_trailing else 0.0
    else:
        trail_trigger = entry_price - trail_start_mult * atr if use_trailing else 0.0

    trailing_on = False
    trail_stop = 0.0
    n = opens.shape[0]

    for j in range(start_i, n):
        # 1) FORCE-END AT LAST BAR INSIDE SESSION
        if j > session_last_i:
            return session_last_i, 3, closes[session_last_i], trailing_on, max_p, min_p

        h = highs[j]
        l = lows[j]
        c = closes[j]

        # track extreme prices
        if h > max_p: max_p = h
        if l < min_p: min_p = l

        # 2) STOP-LOSS
        sl_price = entry_price - sl_mult * atr if side_code == 1 else entry_price + sl_mult * atr
        if (side_code == 1 and l <= sl_price) or (side_code == 2 and h >= sl_price):
            return j, 0, sl_price, False, max_p, min_p

        # 3) TAKE-PROFIT
        tp_price = entry_price + tp_mult * atr if side_code == 1 else entry_price - tp_mult * atr
        if (side_code == 1 and h >= tp_price) or (side_code == 2 and l <= tp_price):
            return j, 1, tp_price, False, max_p, min_p

        # 4) TRAILING-STOP (only if enabled)
        if use_trailing:
            if not trailing_on:
                if (side_code == 1 and h >= trail_trigger) or (side_code == 2 and l <= trail_trigger):
                    trailing_on = True
                    trail_stop = entry_price
            else:
                if side_code == 1:
                    new_trail = max_p - trail_stop_mult * atr
                    if new_trail > trail_stop: trail_stop = new_trail
                    if l <= trail_stop:
                        return j, 2, trail_stop, True, max_p, min_p
                else:
                    new_trail = min_p + trail_stop_mult * atr
                    if new_trail < trail_stop: trail_stop = new_trail
                    if h >= trail_stop:
                        return j, 2, trail_stop, True, max_p, min_p

    # 5) FINAL FALLBACK
    return n - 1, 4, closes[-1], trailing_on, max_p, min_p


def evaluate_crypto_classification(
    X_test,
    preds_stack,
    labeled,
    avoid_funcs,
    TRAIL_START_MULT,
    TRAIL_STOP_MULT,
    initial_balance=100.0,
    leverage=3.0,
    commission_pct=0.0004,  # 0.04% taker fee (Binance futures)
    SL_ATR_MULT=1.5,
    TP_ATR_MULT=2.0,
    RISK_FRACTION=0.02,  # This is now POSITION_FRACTION when using leverage
    MAX_POSITION_PCT=1.0,  # Maximum position as multiple of balance (1.0 = 100%)
    MIN_POSITION_PCT=0.01,  # Minimum position as percentage of balance
    DEBUG=False
):
    """
    Simplified crypto futures backtest with proper leverage and funding rate implementation.

    Key changes:
    - Leverage multiplies your buying power
    - Position sizing based on percentage of leveraged balance
    - Simple commission structure
    - Funding rate payments/receipts calculated at 8-hour intervals.
    - Maintains compatibility with your scoring system
    """
    current_balance = float(initial_balance)
    original_balance = float(initial_balance)
    temp_trades_data = []
    skipped_trades, long_trades, short_trades = 0, 0, 0
    avoid_hits = defaultdict(int)

    # Data Preparation
    index_to_loc = {idx: loc for loc, idx in enumerate(labeled.index)}
    opens = labeled['open'].to_numpy()
    highs = labeled['high'].to_numpy()
    lows = labeled['low'].to_numpy()
    closes = labeled['close'].to_numpy()
    atrs = labeled['ATR_14_5min'].to_numpy()
    times = list(labeled.index)
    preds = np.asarray(preds_stack)
    X_idxs = list(X_test.index)
    
    # Add this line to get funding rates
    funding_rates = labeled['funding_rate'].to_numpy() if 'funding_rate' in labeled.columns else np.zeros_like(opens)


    session_ids = np.empty(len(times), dtype=np.int64)
    if not labeled.empty:
        for k, ts in enumerate(times):
            ts_e = ts
            session_date = ts_e.date() if ts_e.hour >= 18 else (ts_e - pd.Timedelta(days=1)).date()
            session_ids[k] = session_date.year * 10000 + session_date.month * 100 + session_date.day

    i = 0
    while i < len(X_idxs):
        if current_balance <= 0:
            if DEBUG:
                print("Account balance depleted. Stopping simulation.")
            break

        idx = X_idxs[i]
        loc = index_to_loc.get(idx)
        if loc is None:
            i += 1
            continue

        entry_loc = loc + 1
        if entry_loc >= len(opens):
            skipped_trades += 1
            i += 1
            continue

        if times[entry_loc].hour == 17:
            skipped_trades += 1
            i += 1
            continue

        entry_price = opens[entry_loc]
        entry_time = times[entry_loc]
        atr = atrs[loc]
        pred = preds[i]

        if pred == 1:
            side, side_code = 'long', 1
            long_trades += 1
        elif pred == 2:
            side, side_code = 'short', 2
            short_trades += 1
        else:
            skipped_trades += 1
            i += 1
            continue
         
        row = labeled.iloc[loc]
        skip = False
        for name, f in avoid_funcs.items():
            if f(row):
                avoid_hits[name] += 1
                skip = True
        if skip:
            skipped_trades += 1
            i += 1
            continue

        if entry_price <= 0 or atr <= 0 or SL_ATR_MULT <= 0:
            skipped_trades += 1
            i += 1
            continue

        # SIMPLIFIED POSITION SIZING WITH LEVERAGE
        position_fraction = RISK_FRACTION
         
        position_value = current_balance * position_fraction * leverage
         
        max_position_value = current_balance * MAX_POSITION_PCT * leverage
        position_value = min(position_value, max_position_value)
         
        min_position_value = current_balance * MIN_POSITION_PCT
        if position_value < min_position_value:
            skipped_trades += 1
            i += 1
            continue
         
        position_size = position_value / entry_price
         
        margin_used = position_value / leverage

        entry_sess = session_ids[entry_loc]
        session_last_i = next((j - 1 for j in range(entry_loc + 1, len(times)) 
                              if session_ids[j] != entry_sess), len(times) - 1)

        exit_i, reason_code, exit_price, used_trailing, max_price, min_price = simulate_trade_crypto(
            entry_loc + 1, session_last_i, side_code, entry_price, atr,
            SL_ATR_MULT, TP_ATR_MULT, TRAIL_START_MULT, TRAIL_STOP_MULT,
            opens, highs, lows, closes
        )
         
        exit_time = times[exit_i]
        exit_reason = {0: 'SL', 1: 'TP', 2: 'TRAIL', 3: 'SESSION_END', 4: 'FORCED_END'}[reason_code]

        if side == 'long':
            gross_pnl = (exit_price - entry_price) * position_size
        else:
            gross_pnl = (entry_price - exit_price) * position_size
         
        entry_value = position_size * entry_price
        exit_value = position_size * exit_price
        commission = (entry_value * commission_pct) + (exit_value * commission_pct)
        
        # --- NEW: FUNDING RATE CALCULATION ---
        funding_fee = 0
        current_loc = entry_loc
        while current_loc <= exit_i:
            current_time = times[current_loc]
            if current_time.hour % 8 == 0 and current_time.minute == 0:
                # Funding rate is applied to the notional position value
                funding_payment = position_value * funding_rates[current_loc]
                if side == 'long':
                    funding_fee -= funding_payment # Longs pay if funding rate is positive
                else: #short
                    funding_fee += funding_payment # Shorts receive if funding rate is positive
            current_loc += 1
        # --- END OF NEW CODE ---
         
        # Net P&L
        pnl = gross_pnl - commission - funding_fee
         
        balance_before_trade = current_balance
        current_balance += pnl
        current_balance = max(current_balance, 0.01)
         
        if side == 'long':
            mfe = (max_price - entry_price) * position_size
            mae = (entry_price - min_price) * position_size
        else:
            mfe = (entry_price - min_price) * position_size
            mae = (max_price - entry_price) * position_size

        temp_trades_data.append({
            'entry_time': entry_time,
            'exit_time': exit_time,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'mfe': mfe,
            'mae': mae,
            'gross_pnl': gross_pnl,
            'commission': commission,
            'funding_fee': funding_fee, # Added for analysis
            'position_size': position_size,
            'position_value': position_value,
            'margin_used': margin_used,
            'leverage': leverage,
            'exit_reason': exit_reason,
            'used_trailing': used_trailing,
            'balance_before_trade': balance_before_trade,
            'balance_after_trade': current_balance,
        })
         
        if DEBUG and len(temp_trades_data) % 100 == 0:
            last_trade = temp_trades_data[-1]
            print(f"Trade {len(temp_trades_data)}: "
                  f"{last_trade['side']} "
                  f"Position: ${last_trade['position_value']:.2f} "
                  f"(Margin: ${last_trade['margin_used']:.2f}) "
                  f"P&L: ${last_trade['pnl']:.2f} "
                  f"Balance: ${current_balance:.2f}")
         
        while i < len(X_idxs) and times[index_to_loc.get(X_idxs[i], -1)] <= exit_time:
            i += 1

    results = pd.DataFrame(temp_trades_data)
    trades = len(results)
     
    if trades == 0:
        return {
            'pnl': 0, 'trades': 0, 'win_rate': 0, 'expectancy': 0,
            'profit_factor': 0, 'sharpe': 0, 'long_trades': long_trades,
            'short_trades': short_trades, 'avoid_hits': dict(avoid_hits),
            'results': results, 'final_balance': current_balance
        }
     
    pnl_total = results['pnl'].sum()
    win_rate = (results['pnl'] > 0).mean()
    expectancy = results['pnl'].mean()
     
    winning_trades = results.loc[results['pnl'] > 0, 'pnl'].sum()
    losing_trades = results.loc[results['pnl'] < 0, 'pnl'].sum()
     
    if losing_trades != 0:
        profit_factor = abs(winning_trades / losing_trades)
    else:
        profit_factor = np.inf if winning_trades > 0 else 0
     
    if trades > 1:
        returns = results['pnl'] / results['balance_before_trade']
        returns_std = returns.std()
        if returns_std > 0:
            sharpe = (returns.mean() / returns_std) * np.sqrt(252 * 24 * 12)  # Annualized for 5min data
        else:
            sharpe = 0
    else: 
        sharpe = 0

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
        'results': results,
        'final_balance': current_balance
    }
# ────────────────────────────────────────────────────────────────────────────────
# 2) MAIN EVALUATION, ONE TRADE PER SESSION, NO ENTRIES 17:00–18:00 ET
@njit
def simulate_trade(
    start_i,
    session_last_i,
    side_code,
    entry_price,
    atr,
    sl_mult,
    tp_mult,
    trail_start_mult,
    trail_stop_mult,
    opens,
    highs,
    lows,
    closes
):
    max_p = entry_price
    min_p = entry_price

    # only enable trailing when both multipliers > 0
    use_trailing = (trail_start_mult > 0.0) and (trail_stop_mult > 0.0)
    if side_code == 1:
        trail_trigger = entry_price + trail_start_mult * atr if use_trailing else 0.0
    else:
        trail_trigger = entry_price - trail_start_mult * atr if use_trailing else 0.0

    trailing_on = False
    trail_stop  = 0.0
    n = opens.shape[0]

    for j in range(start_i, n):
        # 1) FORCE‐END AT LAST BAR INSIDE SESSION
        if j > session_last_i:
            return session_last_i, 3, closes[session_last_i], trailing_on, max_p, min_p

        h = highs[j]
        l = lows[j]
        c = closes[j]

        # track extreme prices
        if h > max_p: max_p = h
        if l < min_p: min_p = l

        # 2) STOP‐LOSS
        sl_price = entry_price - sl_mult * atr if side_code == 1 else entry_price + sl_mult * atr
        if (side_code == 1 and l <= sl_price) or (side_code == 2 and h >= sl_price):
            return j, 0, sl_price, False, max_p, min_p

        # 3) TAKE‐PROFIT
        tp_price = entry_price + tp_mult * atr if side_code == 1 else entry_price - tp_mult * atr
        if (side_code == 1 and h >= tp_price) or (side_code == 2 and l <= tp_price):
            return j, 1, tp_price, False, max_p, min_p

        # 4) TRAILING‐STOP (only if enabled)
        if use_trailing:
            if not trailing_on:
                if (side_code == 1 and h >= trail_trigger) or (side_code == 2 and l <= trail_trigger):
                    trailing_on = True
                    trail_stop   = entry_price
            else:
                if side_code == 1:
                    new_trail = max_p - trail_stop_mult * atr
                    if new_trail > trail_stop: trail_stop = new_trail
                    if l <= trail_stop:
                        return j, 2, trail_stop, True, max_p, min_p
                else:
                    new_trail = min_p + trail_stop_mult * atr
                    if new_trail < trail_stop: trail_stop = new_trail
                    if h >= trail_stop:
                        return j, 2, trail_stop, True, max_p, min_p

    # 5) FINAL FALLBACK
    return n - 1, 4, closes[-1], trailing_on, max_p, min_p


def evaluate_classification(
    X_test,
    preds_stack,
    labeled,
    avoid_funcs,
    TRAIL_START_MULT,
    TRAIL_STOP_MULT,
    TICK_VALUE,
    SL_ATR_MULT=1.5,
    TP_ATR_MULT=2.0,
    base_contracts=1,
):
    temp_trades_data = []
    skipped_trades  = 0
    avoid_hits      = defaultdict(int)
    long_trades     = 0
    short_trades    = 0

    # fast lookup & arrays
    index_to_loc = {idx: loc for loc, idx in enumerate(labeled.index)}
    opens   = labeled['open'].to_numpy()
    highs   = labeled['high'].to_numpy()
    lows    = labeled['low'].to_numpy()
    closes  = labeled['close'].to_numpy()
    atrs    = labeled['ATR_14_5min'].to_numpy()
    times   = list(labeled.index)
    preds   = np.asarray(preds_stack)
    X_idxs  = list(X_test.index)

    # compute session IDs (6 PM ET → 5 PM ET next day)
    session_ids = np.empty(len(times), dtype=np.int64)
    for k, ts in enumerate(times):
        ts_e = ts.tz_convert('US/Eastern')
        if ts_e.hour >= 18:
            session_date = ts_e.date()
        else:
            session_date = (ts_e - pd.Timedelta(days=1)).date()
        session_ids[k] = (session_date.year * 10000 +
                          session_date.month * 100 +
                          session_date.day)

    i = 0
    while i < len(X_idxs):
        idx = X_idxs[i]
        loc = index_to_loc[idx]

        # ensure there's a next bar
        entry_loc = loc + 1
        if entry_loc >= len(opens):
            skipped_trades += 1
            i += 1
            continue

        # skip entries between 17:00 and 18:00 ET
        entry_bar = times[entry_loc].tz_convert("US/Eastern")
        if entry_bar.hour == 17:
            skipped_trades += 1
            i += 1
            continue

        # clear entry price/time at the open of the next bar
        entry_price = opens[entry_loc]
        entry_time  = times[entry_loc]

        # fetch ATR and prediction
        atr  = atrs[loc]
        pred = preds[i]

        if   pred == 1:
            side, side_code = 'long',  1
            long_trades   += 1
        elif pred == 2:
            side, side_code = 'short', 2
            short_trades  += 1
        else:
            skipped_trades += 1
            i += 1
            continue

        # avoid filters (single-row pandas)
        row = labeled.iloc[loc]
        skip = False
        for name, f in avoid_funcs.items():
            try:
                if f(row):
                    avoid_hits[name] += 1
                    skip = True
            except:
                pass
        if skip:
            skipped_trades += 1
            i += 1
            continue

        # sanity checks
        if entry_price <= 0 or atr <= 0:
            skipped_trades += 1
            i += 1
            continue

        # find the last index *inside* this session
        entry_sess     = session_ids[entry_loc]
        session_last_i = len(times) - 1
        for j in range(entry_loc + 1, len(times)):
            if session_ids[j] != entry_sess:
                session_last_i = j - 1
                break

        # run the compiled forward‐scan
        exit_i, reason_code, exit_price, used_trailing, max_price, min_price = simulate_trade(
            entry_loc + 1,
            session_last_i,
            side_code,
            entry_price,
            atr,
            SL_ATR_MULT,
            TP_ATR_MULT,
            TRAIL_START_MULT,
            TRAIL_STOP_MULT,
            opens,
            highs,
            lows,
            closes
        )
        exit_time = times[exit_i]
        reason_map = {0: 'SL', 1: 'TP', 2: 'TRAIL', 3: 'SESSION_END', 4: 'FORCED_END'}
        exit_reason = reason_map[reason_code]

        # compute PnL, MFE, MAE
        size       = base_contracts
        gross_pnl  = ((exit_price - entry_price) if side == 'long'
                      else (entry_price - exit_price)) * TICK_VALUE * size
        commission = 3.98 * size
        pnl        = gross_pnl - commission

        mfe = ((max_price - entry_price) if side == 'long'
               else (entry_price - min_price)) * TICK_VALUE
        mae = ((entry_price - min_price) if side == 'long'
               else (max_price - entry_price)) * TICK_VALUE

        temp_trades_data.append({
            'entry_time':    entry_time,
            'exit_time':     exit_time,
            'side':          side,
            'entry_price':   entry_price,
            'exit_price':    exit_price,
            'pnl':           pnl,
            'mfe':           mfe,
            'mae':           mae,
            'gross_pnl':     gross_pnl,
            'position_size': size,
            'exit_reason':   exit_reason,
            'used_trailing': used_trailing,
        })

        # advance past this trade’s exit
        while i < len(X_idxs) and times[index_to_loc[X_idxs[i]]] <= exit_time:
            i += 1

    # ────────────────────────────────────────────────────────────────────────────
    # Aggregate stats (unchanged)
    results     = pd.DataFrame(temp_trades_data)
    trades      = len(results)
    pnl_total   = results['pnl'].sum() if trades else 0
    win_rate    = (results['pnl'] > 0).mean() if trades else 0
    expectancy  = results['pnl'].mean()    if trades else 0
    profit_factor = (
        results.loc[results['pnl']>0, 'pnl'].sum()
        / abs(results.loc[results['pnl']<0, 'pnl'].sum())
        if trades and (results['pnl']<0).any() else np.nan
    )
    sharpe = (
        (results['pnl'].mean() / (results['pnl'].std() + 1e-9))
        * np.sqrt(trades) if trades > 1 else 0
    )

    return {
        'pnl':           pnl_total,
        'trades':        trades,
        'win_rate':      win_rate,
        'expectancy':    expectancy,
        'profit_factor': profit_factor,
        'sharpe':        sharpe,
        'long_trades':   long_trades,
        'short_trades':  short_trades,
        'avoid_hits':    dict(avoid_hits),
        'results':       results,
    }

def evaluate_combo(
    X_test, preds_reg_stack, preds_reg_cnn,
    probs_class_stack, probs_class_xgboost,
    le,labeled, df,
    avoid_funcs,
    SL_ATR_MULT, TP_ATR_MULT, TRAIL_START_MULT, TRAIL_STOP_MULT, TICK_VALUE,
    is_same_session,
    long_threshold,
    short_threshold,
    base_contracts=1,
    max_contracts=5,
    skip_weak_conf=False,
    weak_conf_zscore=0.2,
    reg_weights=(0.5, 0.5),  # stack, cnn, lgbm
    class_weights=(0.6, 0.4)     # stack, xgboost
):
    temp_trades_data = []
    skipped_trades = 0
    avoid_hits = defaultdict(int)
    long_trades = 0
    short_trades = 0

    X_test_idx = X_test.index.to_list()

    # Regression ensemble
    reg_ensemble = (
        reg_weights[0] * np.array(preds_reg_stack) +
        reg_weights[1] * np.array(preds_reg_cnn)
    )

    # === Classification ensemble using predicted probabilities ===
    combined_probs = class_weights[0] * probs_class_stack + class_weights[1] * probs_class_xgboost

    # Ensure shape is correct
    if combined_probs.ndim != 2:
        raise ValueError(f"Expected 2D classification probabilities, got shape: {combined_probs.shape}")

    # Predict class index (e.g. 0–4)
    pred_class_idx = np.argmax(combined_probs, axis=1)

    # Decode back to actual labels: [-2, -1, 0, 1, 2]
    pred_class = le.inverse_transform(pred_class_idx)

    # === Confidence from max prob
    conf_scores = np.max(combined_probs, axis=1)
    position_sizes = base_contracts + (max_contracts - base_contracts) * conf_scores
    position_sizes = np.round(position_sizes).astype(int)

    # === Index length check
    X_test_idx = X_test.index.to_list()
    if len(pred_class) != len(X_test_idx):
        raise ValueError(f"Prediction length mismatch: got {len(pred_class)} vs {len(X_test_idx)}")


    for i, idx in enumerate(X_test_idx):
        if idx not in labeled.index or idx + 1 >= len(df):
            skipped_trades += 1
            continue

        reg_pred = reg_ensemble[i]
        row = labeled.loc[idx]
        class_label = pred_class[i]
        conf = conf_scores[i]
        size = position_sizes[i]

        # if skip_weak_conf and conf < weak_conf_zscore:
        #     skipped_trades += 1
        #     continue

        if any(f(row) for name, f in avoid_funcs.items()):
            skipped_trades += 1
            continue

        # 1. Classifier decides direction (but don't update counters yet)
        if class_label > 0:
            side = 'long'
        elif class_label < 0:
            side = 'short'
        else:
            skipped_trades += 1
            continue

        # 2. Check if regression agrees
        reg_agrees = (side == 'long' and reg_pred > long_threshold) or (side == 'short' and reg_pred < short_threshold)
        if not reg_agrees:
            skipped_trades += 1
            continue

        # 3. Now it's safe to count the trade
        if side == 'long':
            long_trades += 1
        else:
            short_trades += 1

        entry_price = df.loc[idx + 1, 'open']
        entry_time = df.loc[idx + 1, 'datetime']
        atr = row['ATR_14_5min']

        expected_move = abs(reg_pred) * entry_price
        tp_move = np.clip(expected_move, 0.001 * entry_price, TP_ATR_MULT * atr)

        # Ensure SL distance is never greater than TP move
        sl_move = min(SL_ATR_MULT * atr, tp_move)

        # Prices
        sl_price = entry_price - sl_move if side == 'long' else entry_price + sl_move
        tp_price = entry_price + tp_move if side == 'long' else entry_price - tp_move
        trail_trigger = entry_price + TRAIL_START_MULT * atr if side == 'long' else entry_price - TRAIL_START_MULT * atr


        max_price, min_price = entry_price, entry_price
        exit_price, exit_time = None, None
        trail_stop = None

        fwd_idx = idx + 1
        while fwd_idx < len(df):
            fwd_row = df.loc[fwd_idx]
            max_price = max(max_price, fwd_row['high'])
            min_price = min(min_price, fwd_row['low'])

            if (side == 'long' and fwd_row['low'] <= sl_price) or (side == 'short' and fwd_row['high'] >= sl_price):
                exit_price = sl_price
                exit_time = fwd_row['datetime']
                break

            if (side == 'long' and fwd_row['high'] >= tp_price) or (side == 'short' and fwd_row['low'] <= tp_price):
                exit_price = tp_price
                exit_time = fwd_row['datetime']
                break

            if side == 'long' and fwd_row['high'] >= trail_trigger:
                trail_stop = fwd_row['close'] - TRAIL_STOP_MULT * atr
            if side == 'short' and fwd_row['low'] <= trail_trigger:
                trail_stop = fwd_row['close'] + TRAIL_STOP_MULT * atr

            if trail_stop:
                if (side == 'long' and fwd_row['low'] <= trail_stop) or (side == 'short' and fwd_row['high'] >= trail_stop):
                    exit_price = trail_stop
                    exit_time = fwd_row['datetime']
                    break

            fwd_idx += 1

        if exit_price is None:
            exit_price = df.loc[len(df) - 1, 'close']
            exit_time = df.loc[len(df) - 1, 'datetime']

        if not is_same_session(entry_time, exit_time):
            continue

        gross_pnl = (exit_price - entry_price) * TICK_VALUE * size if side == 'long' else (entry_price - exit_price) * TICK_VALUE * size
        commission = 3.98 * size
        pnl = gross_pnl - commission
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
            'gross_pnl': gross_pnl,
            'reg_pred': reg_pred,
            'class_pred': class_label,
            'confidence': conf,
            'position_size': size,
        })

    results = pd.DataFrame(temp_trades_data)
    pnl_total = results['pnl'].sum() if not results.empty else 0
    trades = len(results)
    win_rate = (results['pnl'] > 0).mean() if not results.empty else 0
    expectancy = results['pnl'].mean() if not results.empty else 0
    profit_factor = results[results['pnl'] > 0]['pnl'].sum() / abs(results[results['pnl'] < 0]['pnl'].sum()) if not results.empty and (results['pnl'] < 0).any() else np.nan
    sharpe = results['pnl'].mean() / (results['pnl'].std() + 1e-9) * np.sqrt(trades) if trades > 1 else 0

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
        'results': results
    }