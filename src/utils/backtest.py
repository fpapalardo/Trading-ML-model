import numpy as np
import pandas as pd
from collections import defaultdict

from collections import defaultdict
import numpy as np
import pandas as pd
from datetime import timedelta

def session_key(ts: pd.Timestamp) -> pd.Timestamp:
    # shift back 18 h, then floor to midnight to get a unique session “date”
    return (ts - timedelta(hours=18)).normalize()

def is_same_session(start_time: pd.Timestamp, end_time: pd.Timestamp) -> bool:
    return session_key(start_time) == session_key(end_time)

def evaluate_static_tp_two_contracts(
    X_test,
    preds_stack,
    labeled,
    df,
    avoid_funcs,
    TICK_VALUE,
    is_same_session,
    long_thresh,
    short_thresh,
    SL_POINTS: float = 10.0,    # SL in points
    TP_POINTS: float = 10.0     # TP in points for both contracts
):
    """
    2 contracts per signal, both with:
      - Static SL at SL_POINTS
      - Static TP at TP_POINTS
    Only enters trades when predicted move exceeds TP_POINTS.
    """
    temp = []
    skipped = 0
    avoid_hits = defaultdict(int)
    long_cnt = short_cnt = 0

    idxs = X_test.index.to_list()
    preds = preds_stack

    i = 0
    while i < len(idxs):
        idx = idxs[i]
        pos = labeled.index.get_loc(idx)
        if pos + 1 >= len(labeled):
            skipped += 1; i += 1; continue

        row = labeled.iloc[pos]
        entry_row = labeled.iloc[pos + 1]
        entry_price = entry_row["open"]
        entry_time = row.name - pd.Timedelta(minutes=5)
        vol_adj_pred = preds[i]

        # Decide trade direction
        if vol_adj_pred >= long_thresh:
            side = "long"; long_cnt += 1
        elif vol_adj_pred <= short_thresh:
            side = "short"; short_cnt += 1
        else:
            skipped += 1; i += 1; continue

        # Apply avoid filters
        if any(f(row) for f in avoid_funcs.values()):
            for nm, f in avoid_funcs.items():
                try:
                    if f(row):
                        avoid_hits[nm] += 1
                except:
                    pass
            skipped += 1; i += 1; continue

        # Skip if model doesn't predict move > TP_POINTS
        pred_move_pts = abs(vol_adj_pred * entry_price)
        if pred_move_pts < TP_POINTS:
            skipped += 1; i += 1; continue

        sl_price = entry_price - SL_POINTS if side == "long" else entry_price + SL_POINTS
        tp_price = entry_price + TP_POINTS if side == "long" else entry_price - TP_POINTS

        for contract in [1, 2]:
            max_p = min_p = entry_price
            exit_price = exit_time = exit_reason = None
            fwd = pos + 1
            while fwd < len(labeled):
                r = labeled.iloc[fwd]
                hi, lo, cl = r["high"], r["low"], r["close"]
                max_p = max(max_p, hi)
                min_p = min(min_p, lo)

                # SL
                if side == "long" and lo <= sl_price:
                    exit_price, exit_time, exit_reason = sl_price, r.name, f"SL{contract}"; break
                if side == "short" and hi >= sl_price:
                    exit_price, exit_time, exit_reason = sl_price, r.name, f"SL{contract}"; break

                # TP
                if side == "long" and hi >= tp_price:
                    exit_price, exit_time, exit_reason = tp_price, r.name, f"TP{contract}"; break
                if side == "short" and lo <= tp_price:
                    exit_price, exit_time, exit_reason = tp_price, r.name, f"TP{contract}"; break

                # session end
                if not is_same_session(entry_time, r.name):
                    exit_price, exit_time, exit_reason = cl, r.name, f"END{contract}"; break

                fwd += 1

            if exit_price is None:
                rr = labeled.iloc[-1]
                exit_price, exit_time, exit_reason = rr["close"], rr.name, f"FORCE{contract}"

            gross = (exit_price - entry_price) * TICK_VALUE if side == "long" else (entry_price - exit_price) * TICK_VALUE
            pnl = gross - 3.98
            mfe = (max_p - entry_price)*TICK_VALUE if side == "long" else (entry_price - min_p)*TICK_VALUE
            mae = (entry_price - min_p)*TICK_VALUE if side == "long" else (max_p - entry_price)*TICK_VALUE

            temp.append({
                "entry_time": entry_time,
                "exit_time": exit_time,
                "side": side,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "mfe": mfe,
                "mae": mae,
                "gross_pnl": gross,
                "vol_adj_pred": vol_adj_pred,
                "exit_reason": exit_reason,
                "contract": contract
            })

        # skip ahead until after final exit
        while i < len(idxs) and labeled.loc[idxs[i]].name <= exit_time:
            i += 1

    # assemble stats
    df_trades = pd.DataFrame(temp)
    pnl_total = df_trades["pnl"].sum()
    trades = len(df_trades)
    win_rate = (df_trades["pnl"] > 0).mean() if trades > 0 else 0
    expectancy = df_trades["pnl"].mean() if trades > 0 else 0
    profit_factor = (
        df_trades[df_trades["pnl"] > 0]["pnl"].sum() / abs(df_trades[df_trades["pnl"] < 0]["pnl"].sum())
        if (df_trades["pnl"] < 0).any() else np.nan
    )
    sharpe = (
        df_trades["pnl"].mean() / (df_trades["pnl"].std() + 1e-9) * np.sqrt(trades)
        if trades > 1 else 0
    )

    return {
        "pnl": pnl_total,
        "trades": trades,
        "win_rate": win_rate,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "sharpe": sharpe,
        "long_trades": long_cnt,
        "short_trades": short_cnt,
        "avoid_hits": dict(avoid_hits),
        "results": df_trades
    }

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

def evaluate_classification(
    X_test, preds_stack, labeled, df,
    avoid_funcs,
    TRAIL_START_MULT, TRAIL_STOP_MULT, TICK_VALUE,
    SL_ATR_MULT=1.5, TP_ATR_MULT=2.0,
    is_same_session=is_same_session,
    base_contracts=1,
    max_contracts=1,
):
    temp_trades_data = []
    skipped_trades = 0
    avoid_hits = defaultdict(int)
    long_trades = 0
    short_trades = 0

    X_test_idx = X_test.index.to_list()
    # === Combine predicted probabilities ===
    preds_class = np.argmax(preds_stack, axis=1)  # values in {0,1,2}
    conf_scores = np.max(preds_stack, axis=1)
    position_sizes = np.round(
        base_contracts + (max_contracts - base_contracts) * conf_scores, 2
    )

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

        pred        = preds_class[i]
        size         = position_sizes[i]

        # Decide direction
        if pred == 1:
            side = 'long'
            long_trades += 1
        elif pred == 2:
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

        sl_mult = SL_ATR_MULT
        tp_mult = TP_ATR_MULT
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
        'results':              results,
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