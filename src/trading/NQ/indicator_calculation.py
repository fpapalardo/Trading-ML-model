import pandas as pd
import numpy as np
setattr(np, "NaN", np.nan)
import pandas_ta as ta
from datetime import timedelta

# === HELPER FUNCTIONS ===
# === HELPER FUNCTIONS ===
def session_key(ts: pd.Timestamp) -> pd.Timestamp:
    # shift back 18 h, then floor to midnight to get a unique session “date”
    return (ts - timedelta(hours=18)).normalize()

def add_daily_vwap(df, high_col='high', low_col='low', close_col='close', volume_col='volume', new_col_name='VWAP_D'): # Changed name to VWAP_D for daily
    # ... (your existing robust add_daily_vwap function - ensure it uses .copy() and numeric conversions internally)
    # Make sure the final column is named VWAP_D or adjust add_price_vs_ma call later
    if not isinstance(df.index, pd.DatetimeIndex):
        print("Error: DataFrame index must be DatetimeIndex for daily VWAP.")
        return df
    df_temp = df.copy()
    for col in [high_col, low_col, close_col, volume_col]:
        df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
    tpv = ((df_temp[high_col] + df_temp[low_col] + df_temp[close_col]) / 3) * df_temp[volume_col]
    cumulative_tpv = tpv.groupby(df_temp.index.date).cumsum()
    cumulative_volume = df_temp[volume_col].groupby(df_temp.index.date).cumsum()
    vwap_series = cumulative_tpv / cumulative_volume
    df[new_col_name] = vwap_series.replace([np.inf, -np.inf], np.nan)
    return df

def compute_all_indicators(df_input, suffix='', features=None):
    if not isinstance(df_input.index, pd.DatetimeIndex):
        print(f"Warning: DataFrame for suffix '{suffix}' does not have a DatetimeIndex.")

    df = df_input.copy()

    base_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in base_cols):
         raise ValueError(f"DataFrame must contain {base_cols}. Found: {df.columns.tolist()}")
    for col in base_cols: # Ensure correct dtypes for pandas_ta
        df[col] = pd.to_numeric(df[col], errors='coerce')
    #df.dropna(subset=base_cols, inplace=True) # Drop rows if OHLCV became NaN

    if 'atr' in features or 'all' in features:
        df.ta.atr(length=14, append=True, col_names=('ATR_14',))
        df.ta.rsi(length=7, append=True)
        df.ta.rsi(length=14, append=True)

    if 'adx' in features or 'all' in features:
        df.ta.adx(length=14, append=True)
        if 'DMN_14' in df.columns: df.rename(columns={'DMN_14': 'Minus_DI_14'}, inplace=True)

    if 'ema' in features or 'all' in features:
        df.ta.ema(length=20, append=True, col_names=('EMA_20',))
        df.ta.ema(length=50, append=True, col_names=('EMA_50',))
        df.ta.macd(fast=12, slow=26, signal=9, append=True)

    if 'trend' in features or 'all' in features:
        df['Is_Trending'] = (df['ADX_14'] > 20).astype(int)
        df = add_daily_vwap(df, new_col_name='VWAP_D')

    if 'prev_swing' in features or 'all' in features:
        lookback = 20                                              # ≈100 min
        swing_hi_prev = df['high'].rolling(lookback).max().shift(1)
        swing_lo_prev = df['low'].rolling(lookback).min().shift(1)
        swing_mid_prev = 0.5 * (swing_hi_prev + swing_lo_prev)

        df['Price_vs_EMA20'] = (
            (df['close'] - df['EMA_20']) / 
            df['EMA_20'] * 100
        )

        df['Prev_Swing_Dist'] = (
            (df['close'] - swing_mid_prev) / df['ATR_14']
        )

        df['Trend_Alignment'] = (
            (df['EMA_20'] > df['EMA_50']) &  # EMA trend
            (df['close'] > df['VWAP_D']) &            # Above VWAP
            (df['RSI_14'] > 50) &                     # RSI momentum
            (df['MACD_12_26_9'] > 0)                  # MACD positive
        ).astype(int)

        df['Trend_Strength'] = (
            (df['EMA_20'] - df['EMA_50']) / 
            df['ATR_14']
        ).rolling(20).mean()

    if 'volume_trend' in features or 'all' in features:
        df['Trend_Direction'] = np.where(
            df['EMA_20'] > df['EMA_50'], 1, -1
        )

        volume_ma = df['volume'].rolling(20).mean()
        df['Volume_Trend'] = (
            df['volume'] / volume_ma - 1
        ) * df['Trend_Direction']

        df['Trend_Score'] = (
            (df['Trend_Direction'] * 20) +                    # Base direction
            (df['Price_vs_EMA20'].clip(-20, 20)) +          # Price vs EMA
            (df['RSI_14'] - 50) +                           # RSI contribution
            (np.sign(df['MACD_12_26_9']) * 10) +           # MACD direction
            (df['Volume_Trend'].clip(-20, 20)) +           # Volume trend
            (df['Trend_Alignment'] * 20)                    # Alignment bonus
        ).clip(-100, 100)

    if 'poc' in features or 'all' in features:
        df["session_id"] = df.index.map(session_key)
        # 3) compute POC map (approx via bar-close)
        temp = df.assign(price=df['close'])
        vol_profile = temp.groupby(['session_id','price'])['volume'].sum().reset_index()
        poc_map = (
            vol_profile
            .sort_values(['session_id','volume'], ascending=[True,False])
            .drop_duplicates('session_id')
            .set_index('session_id')['price']
        )
        df['POC_Current']  = df['session_id'].map(poc_map)

        df['POC_Dist_Current_Points']  = (df['close'] - df['POC_Current'])

    current_cols = list(df.columns)
    generated_feature_cols = [col for col in current_cols if col not in base_cols]
    rename_dict = {col: col + suffix for col in generated_feature_cols}
    df.rename(columns=rename_dict, inplace=True)

    return df

def session_times(df):
    df['Hour_of_Day'] = df.index.hour
    df['Minute_of_Hour'] = df.index.minute
    df['Day_of_Week'] = df.index.dayofweek
    tf = df['Hour_of_Day'] + df['Minute_of_Hour'] / 60.0
    df['Time_Sin'] = np.sin(2 * np.pi * tf / 24.0)
    df['Day_Sin']  = np.sin(2 * np.pi * df['Day_of_Week'] / 7.0)

    return df
