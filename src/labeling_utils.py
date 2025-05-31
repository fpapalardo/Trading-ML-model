# labeling_utils.py
import os
import numpy as np
import pandas as pd
import numba

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

def compute_regression_labels(
    df: pd.DataFrame, price_col_entry: str = 'open', price_col_exit: str = 'close', lookahead: int = 6,
    vol_col: str = 'ATR_14_5m', min_vol_threshold: float = 0.0001, cap_outliers: bool = True,
    lower_cap_percentile: float = 0.5, upper_cap_percentile: float = 99.5, same_day_trade: bool = True
) -> pd.Series:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be DatetimeIndex.")

    required_cols = [price_col_entry, price_col_exit, vol_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Col '{col}' not found.")
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')

    entry_prices = df[price_col_entry].shift(-1)
    future_prices = df[price_col_exit].shift(-(1 + lookahead))
    volatility = pd.to_numeric(df[vol_col].copy(), errors='coerce')
    volatility[volatility < min_vol_threshold] = min_vol_threshold
    volatility.fillna(min_vol_threshold, inplace=True)

    valid_entry = (entry_prices > 0) & (~entry_prices.isna())
    log_returns = pd.Series(np.nan, index=df.index)
    numeric_future_prices = pd.to_numeric(future_prices[valid_entry], errors='coerce')
    numeric_entry_prices = pd.to_numeric(entry_prices[valid_entry], errors='coerce')

    valid_calc = (numeric_entry_prices > 0) & (numeric_future_prices > 0) & (~numeric_entry_prices.isna()) & (~numeric_future_prices.isna())
    final_calculation_indices = valid_calc[valid_calc].index

    if not final_calculation_indices.empty:
        log_values = np.log(numeric_future_prices.loc[final_calculation_indices] / numeric_entry_prices.loc[final_calculation_indices])
        log_returns.loc[final_calculation_indices] = log_values

    normalized_returns = log_returns / volatility

    if same_day_trade:
        all_dates = pd.Series(df.index.date, index=df.index)
        entry_dates = all_dates.shift(-1)
        exit_dates = all_dates.shift(-(1 + lookahead))
        same_day_mask = (entry_dates == exit_dates)
        normalized_returns.loc[~same_day_mask & normalized_returns.notna()] = np.nan

    if cap_outliers and not normalized_returns.isna().all():
        valid_returns = normalized_returns.dropna()
        if not valid_returns.empty:
            lower_b = np.nanpercentile(valid_returns, lower_cap_percentile)
            upper_b = np.nanpercentile(valid_returns, upper_cap_percentile)
            normalized_returns = normalized_returns.clip(lower=lower_b, upper=upper_b)

    return normalized_returns.rename(f'reg_target_lookahead{lookahead}')

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

    reg_col = f'reg_target_lookahead{lookahead_period}'
    print(f"Adding Regression Target: {reg_col} ...")
    df_labeled[reg_col] = compute_regression_labels(df_labeled, lookahead=lookahead_period, vol_col=vol_col_name)

    print(f"{reg_col} NaNs: {df_labeled[reg_col].isna().sum()}")

    clf_col = f'clf_target_numba_pt{pt_multiplier}sl{sl_multiplier}vb{lookahead_period}'
    print(f"\nAdding Classification Target: {clf_col} ...")
    df_labeled[clf_col] = compute_classification_labels_triple_barrier_numba(
        df_labeled, atr_col=vol_col_name, pt_atr_mult=pt_multiplier,
        sl_atr_mult=sl_multiplier, vertical_barrier_periods=lookahead_period,
        min_target_return_pct=min_return_percentage
    )

    print(f"{clf_col} NaNs: {df_labeled[clf_col].isna().sum()}")

    df_final = df_labeled.dropna(subset=[reg_col, clf_col])
    print(f"Rows after dropping NaNs from targets: {len(df_final)}")

    if not df_final.empty:
        os.makedirs("parquet", exist_ok=True)
        output_filename = os.path.join("parquet", f"labeled_data_{output_file_suffix}.parquet")
        df_final.to_parquet(output_filename)
        print(f"✅ Saved {output_filename} with {len(df_final)} rows")
    else:
        print(f"❌ No data left to save for {output_file_suffix} after dropping NaN targets.")

    return df_final
