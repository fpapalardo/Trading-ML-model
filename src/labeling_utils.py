# labeling_utils.py
import os
import numpy as np
import pandas as pd
import numba

def transform_target(y, scale_factor=1000):
    """
    Transform the target variable to handle small values better
    """
    # Apply sigmoid-like transformation while preserving sign
    return np.sign(y) * np.tanh(np.abs(y) * scale_factor)

@numba.jit(nopython=True, nogil=True)
def _calculate_triple_barrier_labels_numba_core(
    entry_prices_arr: np.ndarray, high_prices_arr: np.ndarray, low_prices_arr: np.ndarray, atr_arr: np.ndarray,
    pt_atr_mult: float, sl_atr_mult: float, vertical_barrier_periods: int, min_target_return_pct: float
) -> np.ndarray:
    n = len(entry_prices_arr)
    labels = np.zeros(n, dtype=np.int32)
    for i in range(n - vertical_barrier_periods):
        entry_price = entry_prices_arr[i]
        current_atr = atr_arr[i]
        if np.isnan(entry_price) or np.isnan(current_atr) or current_atr == 0:
            labels[i] = 0
            continue

        long_tp = entry_price + (current_atr * pt_atr_mult)
        long_sl = entry_price - (current_atr * sl_atr_mult)
        long_result = 0

        for j in range(vertical_barrier_periods):
            future_idx = i + 1 + j
            if future_idx >= n: break
            if high_prices_arr[future_idx] >= long_tp:
                long_result = 1
                break
            if low_prices_arr[future_idx] <= long_sl:
                long_result = -1
                break

        short_tp = entry_price - (current_atr * pt_atr_mult)
        short_sl = entry_price + (current_atr * sl_atr_mult)
        short_result = 0

        for j in range(vertical_barrier_periods):
            future_idx = i + 1 + j
            if future_idx >= n: break
            if low_prices_arr[future_idx] <= short_tp:
                short_result = 1
                break
            if high_prices_arr[future_idx] >= short_sl:
                short_result = -1
                break

        if long_result == 1 and short_result != 1:
            labels[i] = 1
        elif short_result == 1 and long_result != 1:
            labels[i] = 2
        else:
            labels[i] = 0
    return labels

def compute_classification_labels_triple_barrier_numba(
    df_prices: pd.DataFrame, entry_price_col='close', high_col='high', low_col='low',
    atr_col='ATR_14_5m', pt_atr_mult=2.0, sl_atr_mult=1.5,
    vertical_barrier_periods=10, min_target_return_pct=0.0005
) -> pd.Series:
    if not isinstance(df_prices.index, pd.DatetimeIndex):
        raise ValueError("Index must be DatetimeIndex.")

    for col in [entry_price_col, high_col, low_col, atr_col]:
        if col not in df_prices.columns:
            raise KeyError(f"Missing column: {col}")

    entry_prices = df_prices[entry_price_col].to_numpy(dtype=np.float64)
    highs = df_prices[high_col].to_numpy(dtype=np.float64)
    lows = df_prices[low_col].to_numpy(dtype=np.float64)
    atrs = df_prices[atr_col].to_numpy(dtype=np.float64)

    labels = _calculate_triple_barrier_labels_numba_core(
        entry_prices, highs, lows, atrs,
        pt_atr_mult, sl_atr_mult, vertical_barrier_periods, min_target_return_pct
    )
    return pd.Series(labels, index=df_prices.index, dtype=int).rename(
        f'clf_target_numba_pt{pt_atr_mult}sl{sl_atr_mult}vb{vertical_barrier_periods}'
    )

# def compute_regression_labels(
#     df: pd.DataFrame, price_col_entry: str = 'open', price_col_exit: str = 'close', lookahead: int = 6,
#     vol_col: str = 'ATR_14_5m', min_vol_threshold: float = 0.0001, cap_outliers: bool = True,
#     lower_cap_percentile: float = 0.5, upper_cap_percentile: float = 99.5, same_day_trade: bool = True,
#     should_transform=True, scale_factor=1000
# ) -> pd.Series:
#     if not isinstance(df.index, pd.DatetimeIndex):
#         raise ValueError("Index must be DatetimeIndex.")

#     required_cols = [price_col_entry, price_col_exit, vol_col]
#     for col in required_cols:
#         if col not in df.columns:
#             raise ValueError(f"Col '{col}' not found.")
#         if not pd.api.types.is_numeric_dtype(df[col]):
#             df[col] = pd.to_numeric(df[col], errors='coerce')

#     entry_prices = df[price_col_entry].shift(-1)
#     future_prices = df[price_col_exit].shift(-(1 + lookahead))
#     volatility = pd.to_numeric(df[vol_col].copy(), errors='coerce')
#     volatility[volatility < min_vol_threshold] = min_vol_threshold
#     volatility.fillna(min_vol_threshold, inplace=True)

#     valid_entry = (entry_prices > 0) & (~entry_prices.isna())
#     log_returns = pd.Series(np.nan, index=df.index)
#     numeric_future_prices = pd.to_numeric(future_prices[valid_entry], errors='coerce')
#     numeric_entry_prices = pd.to_numeric(entry_prices[valid_entry], errors='coerce')

#     valid_calc = (numeric_entry_prices > 0) & (numeric_future_prices > 0) & (~numeric_entry_prices.isna()) & (~numeric_future_prices.isna())
#     final_calculation_indices = valid_calc[valid_calc].index

#     if not final_calculation_indices.empty:
#         log_values = np.log(numeric_future_prices.loc[final_calculation_indices] / numeric_entry_prices.loc[final_calculation_indices])
#         log_returns.loc[final_calculation_indices] = log_values

#     normalized_returns = log_returns / volatility

#     if same_day_trade:
#         all_dates = pd.Series(df.index.date, index=df.index)
#         entry_dates = all_dates.shift(-1)
#         exit_dates = all_dates.shift(-(1 + lookahead))
#         same_day_mask = (entry_dates == exit_dates)
#         normalized_returns.loc[~same_day_mask & normalized_returns.notna()] = np.nan

#     if cap_outliers and not normalized_returns.isna().all():
#         valid_returns = normalized_returns.dropna()
#         if not valid_returns.empty:
#             lower_b = np.nanpercentile(valid_returns, lower_cap_percentile)
#             upper_b = np.nanpercentile(valid_returns, upper_cap_percentile)
#             normalized_returns = normalized_returns.clip(lower=lower_b, upper=upper_b)

#     if should_transform:
#         normalized_returns = transform_target(normalized_returns, scale_factor)  # Use your existing transform_target function

#     return normalized_returns.rename(f'reg_target_lookahead{lookahead}')

def _compute_session_end_indices(
    dts: np.ndarray, 
    open_hour: int = 18, 
    next_day_cutoff_min: int = 12 * 60
) -> np.ndarray:
    """
    Precompute, for each bar index i, the integer index of the last bar
    we should consider for the “session” that began at i+1.
    If time ≥ 18:00 local, roll to next day’s first bar ≥ 12:00; 
    otherwise use last bar of same day.
    """
    N = len(dts)
    session_end = np.empty(N, dtype=np.int64)

    # Extract date as YYYYMMDD int, and minutes since midnight:
    dates   = np.empty(N, dtype=np.int32)
    minutes = np.empty(N, dtype=np.int32)
    for i in range(N):
        ts = pd.Timestamp(dts[i])
        dates[i]   = ts.year * 10000 + ts.month * 100 + ts.day
        minutes[i] = ts.hour * 60 + ts.minute

    # Map date → last index of that date
    last_of_date = {}
    for i in range(N):
        last_of_date[dates[i]] = i

    # Map date → all indices of that date (sorted by index)
    idxs_by_date = {}
    for i in range(N):
        d = dates[i]
        if d not in idxs_by_date:
            idxs_by_date[d] = []
        idxs_by_date[d].append(i)

    for i in range(N):
        d_i = dates[i]
        m_i = minutes[i]

        if m_i >= open_hour * 60:
            # Build the “next day” key
            y  = d_i // 10000
            mo = (d_i // 100) % 100
            da = d_i % 100
            next_date = (pd.Timestamp(y, mo, da) + pd.Timedelta(days=1)).date()
            next_key = next_date.year * 10000 + next_date.month * 100 + next_date.day

            if next_key not in idxs_by_date:
                # If next day not in data, fallback to last of current day
                session_end[i] = last_of_date[d_i]
            else:
                # Among all bars on next day, pick the first one ≥ 12:00; else last of next day
                cand_idxs = idxs_by_date[next_key]
                chosen = cand_idxs[-1]  # default to last bar of next day
                for c in cand_idxs:
                    if minutes[c] >= next_day_cutoff_min:
                        chosen = c
                        break
                session_end[i] = chosen
        else:
            # Same day → last bar of current day
            session_end[i] = last_of_date[d_i]

    return session_end


@numba.jit(nopython=True)
def _label_loop(
    opens:      np.ndarray,
    lows:       np.ndarray,
    highs:      np.ndarray,
    closes:     np.ndarray,
    atrs:       np.ndarray,
    adxs:       np.ndarray,
    chops:      np.ndarray,
    session_end_idx: np.ndarray,
    N:          int,
    # SL/TP multipliers
    tp_trend:   float,
    sl_trend:   float,
    tp_chop:    float,
    sl_chop:    float,
    tp_def:     float,
    sl_def:     float,
    round_trip_cost: float
):
    """
    Core Numba loop.  Returns two 1D arrays: out_value (net P/L %) 
    and out_side (1 if long, –1 if short).  

    Key point: `round_trip_cost` is the fixed \$3.98 per trade.  
    Inside the loop, we convert it to a percentage of entry_price by dividing
    by (entry_price * 20) because each 1‐point move on NQ = \$20.
    """
    out_value = np.full(N, np.nan, dtype=np.float64)
    out_side  = np.zeros(N, dtype=np.int64)

    for i in range(N - 1):
        idx = i + 1
        entry_price = opens[idx]
        atr = atrs[idx]
        if entry_price <= 0.0 or atr <= 0.0:
            # skip invalid bars
            continue

        # 1) Choose regime at entry
        adx_i  = adxs[idx]
        chop_i = chops[idx]
        if (adx_i > 25.0) and (chop_i < 50.0):
            tp_m = tp_trend
            sl_m = sl_trend
        elif chop_i > 60.0:
            tp_m = tp_chop
            sl_m = sl_chop
        else:
            tp_m = tp_def
            sl_m = sl_def

        # 2) Compute SL/TP prices
        sl_long  = entry_price - sl_m * atr
        tp_long  = entry_price + tp_m * atr
        sl_short = entry_price + sl_m * atr
        tp_short = entry_price - tp_m * atr

        end_idx = session_end_idx[i]

        exit_long  = np.nan
        exit_short = np.nan

        # 3) Scan forward until session_end_idx
        for j in range(idx + 1, end_idx + 1):
            low_j  = lows[j]
            high_j = highs[j]

            if np.isnan(exit_long):
                if low_j <= sl_long:
                    exit_long = (sl_long - entry_price) / entry_price
                elif high_j >= tp_long:
                    exit_long = (tp_long - entry_price) / entry_price

            if np.isnan(exit_short):
                if high_j >= sl_short:
                    exit_short = (entry_price - sl_short) / entry_price
                elif low_j <= tp_short:
                    exit_short = (entry_price - tp_short) / entry_price

            if (not np.isnan(exit_long)) and (not np.isnan(exit_short)):
                break

        # 4) If no SL/TP was hit, use final bar’s close
        if np.isnan(exit_long):
            final_p = closes[end_idx]
            exit_long = (final_p - entry_price) / entry_price

        if np.isnan(exit_short):
            final_p = closes[end_idx]
            exit_short = (entry_price - final_p) / entry_price

        # 5) Subtract fixed \$3.98 cost as a % of entry_price:
        #       cost_pct = (3.98 dollars) ÷ (entry_price * $20/point)
        cost_pct = round_trip_cost / (entry_price * 20.0)
        exit_long  -= cost_pct
        exit_short -= cost_pct

        # 6) Pick whichever side gives the larger (net) P/L
        if exit_long >= exit_short:
            out_value[i] = exit_long
            out_side[i]  = 1
        else:
            out_value[i] = exit_short
            out_side[i]  = -1

    return out_value, out_side


def compute_dynamic_dual_labels_with_regime_and_cost_numba(
    df: pd.DataFrame,
    atr_col: str     = "ATR_14_5min",
    price_col: str   = "open",
    adx_col: str     = "ADX_14_5min",
    chop_col: str    = "CHOP_14_1_100_5min",
    tp_atr_mult_trend: float   = 3.0,
    sl_atr_mult_trend: float   = 1.0,
    tp_atr_mult_chop: float    = 1.0,
    sl_atr_mult_chop: float    = 0.5,
    tp_atr_mult_default: float = 2.0,
    sl_atr_mult_default: float = 1.0,
    round_trip_cost: float     = 3.98,
) -> pd.DataFrame:
    """
    Main entrypoint.  Returns a DataFrame with two columns:
      - 'reg_value' (net P/L % after cost)
      - 'reg_side'  (1 if long > short, –1 otherwise)
    """
    assert isinstance(df.index, pd.DatetimeIndex), "Index must be DatetimeIndex"
    for col in ("low", "high", "close", price_col, atr_col, adx_col, chop_col):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in df")

    N = len(df)
    opens  = df[price_col].to_numpy(dtype=np.float64)
    lows   = df["low"].to_numpy(dtype=np.float64)
    highs  = df["high"].to_numpy(dtype=np.float64)
    closes = df["close"].to_numpy(dtype=np.float64)
    atrs   = df[atr_col].fillna(0.0).to_numpy(dtype=np.float64)
    adxs   = df[adx_col].fillna(0.0).to_numpy(dtype=np.float64)
    chops  = df[chop_col].fillna(0.0).to_numpy(dtype=np.float64)

    # 1) Precompute session_end_idx array
    dts = df.index.values  # np.ndarray of dtype datetime64[ns]
    session_end_idx = _compute_session_end_indices(dts)

    # 2) Run the jitted loop
    vals, sides = _label_loop(
        opens, lows, highs, closes,
        atrs, adxs, chops,
        session_end_idx, N,
        tp_atr_mult_trend, sl_atr_mult_trend,
        tp_atr_mult_chop, sl_atr_mult_chop,
        tp_atr_mult_default, sl_atr_mult_default,
        round_trip_cost     # pass the raw \$3.98 here
    )

    df_labels = pd.DataFrame({
        "reg_value": vals,
        "reg_side":  sides
    }, index=df.index)

    return df_labels


def label_and_save(
    df_input_features: pd.DataFrame, 
    lookahead_period: int,
    vol_col_name: str,
    pt_multiplier: float,
    sl_multiplier: float,
    min_return_percentage: float,
    output_file_suffix: str,
    feature_columns_for_dropna: list
):
    print(f"\n--- Processing for output suffix: {output_file_suffix} ---")
    df_labeled = df_input_features.copy()

    # 1) Regression targets: compute both value and side using the dynamic SL/TP logic
    print("Adding Regression Targets...")
    df_reg = compute_dynamic_dual_labels_with_regime_and_cost_numba(
        df_input_features,
        atr_col="ATR_14_5min",
        price_col="open",
        adx_col="ADX_14_5min",
        chop_col="CHOP_14_1_100_5min",
        tp_atr_mult_trend=3.0,
        sl_atr_mult_trend=1.0,
        tp_atr_mult_chop=1.0,
        sl_atr_mult_chop=0.5,
        tp_atr_mult_default=2.0,
        sl_atr_mult_default=1.0,
        round_trip_cost=3.98,
    )
    # df_reg contains columns ["reg_value", "reg_side"]
    df_labeled["reg_value"] = df_reg["reg_value"]
    df_labeled["reg_side"]  = df_reg["reg_side"]

    print(f"reg_value NaNs: {df_labeled['reg_value'].isna().sum()}")
    print(f"reg_side  NaNs: {df_labeled['reg_side'].isna().sum()}")

    # 2) Classification target (unmodified)
    clf_col = f'clf_target_numba_pt{pt_multiplier}sl{sl_multiplier}vb{lookahead_period}'
    print(f"\nAdding Classification Target: {clf_col} ...")
    df_labeled[clf_col] = compute_classification_labels_triple_barrier_numba(
        df_labeled,
        atr_col=vol_col_name,
        pt_atr_mult=pt_multiplier,
        sl_atr_mult=sl_multiplier,
        vertical_barrier_periods=lookahead_period,
        min_target_return_pct=min_return_percentage
    )
    print(f"{clf_col} NaNs: {df_labeled[clf_col].isna().sum()}")

    # 3) Drop any rows where regression or classification labels are NaN
    required_cols = ["reg_value", "reg_side", clf_col]
    df_final = df_labeled.dropna(subset=required_cols)
    print(f"Rows after dropping NaNs from targets: {len(df_final)}")

    # 4) Save to parquet if any rows remain
    if not df_final.empty:
        os.makedirs("parquet", exist_ok=True)
        output_filename = os.path.join("parquet", f"labeled_data_{output_file_suffix}.parquet")
        df_final.to_parquet(output_filename)
        print(f"✅ Saved {output_filename} with {len(df_final)} rows")
    else:
        print(f"❌ No data left to save for {output_file_suffix} after dropping NaN targets.")

    return df_final