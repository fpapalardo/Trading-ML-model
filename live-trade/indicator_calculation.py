import pandas as pd
import numpy as np
import pandas_ta as ta

# === HELPER FUNCTIONS ===

def add_price_vs_ma(df, price_col='close', ma_col_name='EMA_20', new_col_name_suffix='_vs_EMA20'):
    # Ensure ma_col_name exists (it would have been created by pandas_ta)
    if ma_col_name in df.columns and price_col in df.columns:
        # Ensure inputs are numeric before division
        df[price_col + new_col_name_suffix] = pd.to_numeric(df[price_col], errors='coerce') / pd.to_numeric(df[ma_col_name], errors='coerce')
    return df

def add_ma_vs_ma(df, ma1_col_name='EMA_10', ma2_col_name='EMA_20', new_col_name_suffix='_vs_EMA20'):
    if ma1_col_name in df.columns and ma2_col_name in df.columns:
        df[ma1_col_name + new_col_name_suffix] = pd.to_numeric(df[ma1_col_name], errors='coerce') / pd.to_numeric(df[ma2_col_name], errors='coerce')
    return df

def add_ma_slope(df, ma_col_name='EMA_10', new_col_name_suffix='_Slope_10', periods=1):
    if ma_col_name in df.columns:
        df[new_col_name_suffix] = pd.to_numeric(df[ma_col_name], errors='coerce').diff(periods) / periods
    return df

def add_rsi_signals(df, rsi_col_name='RSI_14', ob_level=70, os_level=30):
    if rsi_col_name in df.columns:
        rsi_series = pd.to_numeric(df[rsi_col_name], errors='coerce')
        df[rsi_col_name + f'_Is_Overbought_{ob_level}'] = (rsi_series > ob_level).astype(int)
        df[rsi_col_name + f'_Is_Oversold_{os_level}'] = (rsi_series < os_level).astype(int)
    return df

def add_stoch_signals(df, stoch_k_col_name='STOCHk_14_3_3', ob_level=80, os_level=20): # Default pandas_ta name for k
    if stoch_k_col_name in df.columns:
        stoch_k_series = pd.to_numeric(df[stoch_k_col_name], errors='coerce')
        df[stoch_k_col_name + f'_Is_Overbought_{ob_level}'] = (stoch_k_series > ob_level).astype(int)
        df[stoch_k_col_name + f'_Is_Oversold_{os_level}'] = (stoch_k_series < os_level).astype(int)
    return df

def add_macd_cross_signal(df, macd_col_name='MACD_12_26_9', signal_col_name='MACDs_12_26_9'): # Default pandas_ta name for signal
    if macd_col_name in df.columns and signal_col_name in df.columns:
        macd_series = pd.to_numeric(df[macd_col_name], errors='coerce')
        signal_series = pd.to_numeric(df[signal_col_name], errors='coerce')
        crossed_above = (macd_series > signal_series) & (macd_series.shift(1) < signal_series.shift(1))
        crossed_below = (macd_series < signal_series) & (macd_series.shift(1) > signal_series.shift(1))
        df[macd_col_name + '_Cross_Signal'] = np.where(crossed_above, 1, np.where(crossed_below, -1, 0))
    return df

def add_price_vs_bb(df, price_col='close', bb_upper_col='BBU_20_2.0', bb_lower_col='BBL_20_2.0'): # Default pandas_ta names
    if price_col in df.columns and bb_upper_col in df.columns and bb_lower_col in df.columns:
        price_series = pd.to_numeric(df[price_col], errors='coerce')
        bb_upper_series = pd.to_numeric(df[bb_upper_col], errors='coerce')
        bb_lower_series = pd.to_numeric(df[bb_lower_col], errors='coerce')
        df[price_col + '_vs_BB_Upper'] = (price_series > bb_upper_series).astype(int)
        df[price_col + '_vs_BB_Lower'] = (price_series < bb_lower_series).astype(int)
    return df

def add_psar_flip_signal(df, psar_col_name='PSARr_0.02_0.2', close_col='close'): # pandas_ta PSAR reversal column
    if psar_col_name in df.columns: # PSARr is 1 for reversal to uptrend, -1 for reversal to downtrend
        df['PSAR_Flip_Signal'] = pd.to_numeric(df[psar_col_name], errors='coerce').fillna(0).astype(int)
    return df

# Keep these custom functions as they are generally good:
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


def add_candle_features(df):
    # ... (your existing add_candle_features function - ensure numeric conversions) ...
    df_temp = df.copy()
    for col in ['open', 'high', 'low', 'close']:
        df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
    df['Candle_Range'] = df_temp['high'] - df_temp['low']
    df['Candle_Body'] = (df_temp['close'] - df_temp['open']).abs()
    df['Upper_Wick'] = df_temp['high'] - np.maximum(df_temp['open'], df_temp['close'])
    df['Lower_Wick'] = np.minimum(df_temp['open'], df_temp['close']) - df_temp['low']
    df['Body_vs_Range'] = (df['Candle_Body'] / df['Candle_Range'].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)
    return df

def add_return_features(df, price_col='close'):
    # ... (your existing add_return_features function - ensure numeric conversions and inf handling) ...
    price_series_num = pd.to_numeric(df[price_col], errors='coerce').replace(0, np.nan)
    df[f'Log_Return_1'] = np.log(price_series_num / price_series_num.shift(1))
    df[f'Log_Return_3'] = np.log(price_series_num / price_series_num.shift(3))
    df[f'Log_Return_6'] = np.log(price_series_num / price_series_num.shift(6))
    df[f'Simple_Return_1'] = price_series_num.pct_change(1)
    for col_ret in [f'Log_Return_1', f'Log_Return_3', f'Log_Return_6', f'Simple_Return_1']:
        if col_ret in df.columns: df[col_ret] = df[col_ret].replace([np.inf, -np.inf], np.nan)
    return df

def add_rolling_stats(df, price_col='close', window1=14, window2=30):
    # ... (your existing add_rolling_stats function - ensure numeric conversions and inf handling) ...
    returns = pd.to_numeric(df[price_col], errors='coerce').pct_change(1).replace([np.inf, -np.inf], np.nan)
    df[f'Rolling_Std_Dev_{window1}'] = returns.rolling(window=window1).std()
    df[f'Rolling_Skew_{window2}'] = returns.rolling(window=window2).skew()
    df[f'Rolling_Kurtosis_{window2}'] = returns.rolling(window=window2).kurt()
    return df

def add_lagged_features(df, cols_to_lag, lags=[1, 3, 6]):
    # ... (your existing add_lagged_features function - ensure numeric conversions on source col if needed) ...
    for col_orig in cols_to_lag:
        if col_orig in df.columns:
            series_to_lag = pd.to_numeric(df[col_orig], errors='coerce')
            for lag in lags:
                df[f'{col_orig}_Lag_{lag}'] = series_to_lag.shift(lag)
    return df

def compute_all_indicators(df_input, suffix='', indicators=None):
    """
    Adds technical indicators and derived features using pandas_ta.
    Assumes df_input has 'open', 'high', 'low', 'close', 'volume' columns (DatetimeIndex).
    """
    if not isinstance(df_input.index, pd.DatetimeIndex):
        print(f"Warning: DataFrame for suffix '{suffix}' does not have a DatetimeIndex.")
    
    df = df_input.copy() # Work on a copy

    # Ensure base OHLCV columns are numeric and present
    base_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in base_cols):
         raise ValueError(f"DataFrame must contain {base_cols}. Found: {df.columns.tolist()}")
    for col in base_cols: # Ensure correct dtypes for pandas_ta
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=base_cols, inplace=True) # Drop rows if OHLCV became NaN

    if df.empty:
        print(f"DataFrame became empty after coercing OHLCV for suffix '{suffix}'. Returning empty DataFrame.")
        # Return an empty dataframe with expected suffixed columns if possible or raise error
        # For simplicity, we'll let it create columns that will be all NaN, then suffixing will apply.
        # Or handle more gracefully by creating expected columns with NaNs.
        # For now, we will proceed, and suffixing will apply to what gets created.
        pass

    # I. Technical Indicators using pandas_ta
    # Most pandas_ta functions automatically name columns (e.g., SMA_10, RSI_14)
    # and handle NaNs internally. `append=True` adds them to df.

    # Volume
    if "Volume_SMA" in indicators:
        df.ta.sma(close=df['volume'], length=20, append=True, col_names=('Volume_SMA_20'))
    
    if "VWAP" in indicators:
        df = add_daily_vwap(df, new_col_name='VWAP_D') # Using your custom daily VWAP, named VWAP_D
        df = add_price_vs_ma(df, ma_col_name='VWAP_D', new_col_name_suffix='_vs_VWAP_D')

    # Volatility
    if "BB" in indicators:
        df.ta.bbands(length=20, std=2, append=True) # Creates BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0
        # Helpers will need these names: BBU_20_2.0, BBL_20_2.0
        df = add_price_vs_bb(df, bb_upper_col='BBU_20_2.0', bb_lower_col='BBL_20_2.0')

    if "ATR" in indicators:
        df.ta.atr(length=14, append=True, col_names=('ATR_14')) # pandas_ta might name it ATRr_14 or similar. We force ATR_14.

    # Trend
    if "SMA" in indicators:
        df.ta.sma(length=10, append=True) # SMA_10
        df.ta.sma(length=20, append=True) # SMA_20 (also BBM_20_2.0 from bbands)
        df.ta.sma(length=50, append=True) # SMA_50

    if "EMA" in indicators:
        df.ta.ema(length=10, append=True) # EMA_10
        df.ta.ema(length=20, append=True) # EMA_20
        df.ta.ema(length=50, append=True) # EMA_50
    
    if "SLOPE" in indicators:
        df = add_price_vs_ma(df, ma_col_name='EMA_20', new_col_name_suffix='_vs_EMA20')
        df = add_ma_vs_ma(df, ma1_col_name='EMA_10', ma2_col_name='EMA_20', new_col_name_suffix='_vs_EMA20')
        df = add_ma_slope(df, ma_col_name='EMA_10', new_col_name_suffix='_Slope_10')

    if "MACD" in indicators:
        df.ta.macd(fast=12, slow=26, signal=9, append=True) # MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
        df = add_macd_cross_signal(df, macd_col_name='MACD_12_26_9', signal_col_name='MACDs_12_26_9')

    if "ADX" in indicators:
        df.ta.adx(length=14, append=True) # ADX_14, DMP_14, DMN_14
    # Rename DMN_14 and DMP_14 to match your old Minus_DI_14, Plus_DI_14 if helpers depend on it
    if 'DMP_14' in df.columns: df.rename(columns={'DMP_14': 'Plus_DI_14'}, inplace=True)
    if 'DMN_14' in df.columns: df.rename(columns={'DMN_14': 'Minus_DI_14'}, inplace=True)
    
    if "PSAR" in indicators:
        df.ta.psar(append=True) # Creates PSARl_0.02_0.2, PSARs_0.02_0.2, PSARaf_0.02_0.2, PSARr_0.02_0.2
        df = add_psar_flip_signal(df, psar_col_name='PSARr_0.02_0.2') # Use reversal column

    if "CCI" in indicators:
        df.ta.cci(length=20, append=True, col_names=('CCI_20')) # pandas_ta uses CCI_20_0.015 by default

    # Momentum
    if "RSI" in indicators:
        df.ta.rsi(length=14, append=True) # RSI_14
        df = add_rsi_signals(df, rsi_col_name='RSI_14')
    
    if "STOCHk" in indicators:
        df.ta.stoch(k=14, d=3, smooth_k=3, append=True) # STOCHk_14_3_3, STOCHd_14_3_3
        df = add_stoch_signals(df, stoch_k_col_name='STOCHk_14_3_3')

    if "PPO" in indicators:
        df.ta.ppo(fast=12, slow=26, signal=9, append=True) # PPO_12_26_9, PPOh_12_26_9, PPOs_12_26_9
        df.ta.roc(length=10, append=True) # ROC_10
    
    # Explicitly convert PPO and ROC to numeric (belt and braces after pandas_ta)
    for col_name in ['PPO_12_26_9', 'ROC_10']: # Check exact names if pandas_ta produces variants
        base_name_ppo = [c for c in df.columns if "PPO_" in c and "PPOh" not in c and "PPOs" not in c]
        base_name_roc = [c for c in df.columns if "ROC_" in c]
        
        for actual_col_name in base_name_ppo + base_name_roc:
            if actual_col_name in df.columns:
                df[actual_col_name] = df[actual_col_name].replace([np.inf, -np.inf], np.nan)
                df[actual_col_name] = pd.to_numeric(df[actual_col_name], errors='coerce')

    # II. Price Action & Basic Features (Keep your custom functions)
    if "CDL" in indicators:
        df = add_candle_features(df)
        # df = add_candlestick_patterns(df) # We'll replace this with pandas_ta candlestick patterns

        # --- pandas_ta Candlestick Patterns ---
        # Example: Add Doji, Hammer, Engulfing. pandas_ta has many more.
        # 'name="all"' would add many columns, so be selective or use a list.
        # === Candlestick Patterns (optional, may return None if insufficient data)
        candle_patterns_to_check = ["doji", "hammer", "engulfing"]

        for pattern in candle_patterns_to_check:
            try:
                pattern_df = df.ta.cdl_pattern(name=pattern)
                if pattern_df is not None and not pattern_df.empty:
                    original_col = pattern_df.columns[0]
                    df[original_col] = pattern_df[original_col]

                    # 👇 Rename to your preferred format
                    if pattern.lower() == "doji":
                        new_col = "CDL_DOJI_10_0.1"
                    else:
                        new_col = f"CDL_{pattern.upper()}"

                    df.rename(columns={original_col: new_col}, inplace=True)
                else:
                    print(f"[Candlestick] Pattern '{pattern}' returned None or empty, skipping.")
            except Exception as e:
                print(f"[Candlestick] Error with pattern '{pattern}': {e}")

    # Rename columns to match your old convention if needed, e.g., CDLDOJI -> Is_Doji
    if 'CDLDOJI' in df.columns: df.rename(columns={'CDLDOJI': 'Is_Doji_pta'}, inplace=True) # Add _pta to distinguish
    if 'CDLHAMMER' in df.columns: df.rename(columns={'CDLHAMMER': 'Is_Hammer_pta'}, inplace=True)
    if 'CDLENGULFING' in df.columns: df.rename(columns={'CDLENGULFING': 'Is_Engulfing_pta'}, inplace=True) # This is a general engulfing signal (+/-)

    if "RETURN" in indicators:
        df = add_return_features(df)

    # III. Statistical Features (Keep your custom functions)
    if "ROLLING" in indicators:
        df = add_rolling_stats(df)
    
    # Lagged Features
    # Ensure base columns for lagging are the ones created by pandas_ta or your helpers
    cols_to_lag_pta = ['close', 'RSI_14', 'Candle_Body', 'Volume_SMA_20'] 
    # Check if these columns actually exist, as pandas_ta might name them slightly differently
    # This valid_cols_to_lag should use the names as they are in df at this point
    valid_cols_to_lag = [col for col in cols_to_lag_pta if col in df.columns]
    if "LAG" in indicators:
        df = add_lagged_features(df, valid_cols_to_lag, lags=[1,2,3])

    # --- Suffixing ---
    # All columns created by pandas_ta (that were appended) or by helpers
    # that are not the original 'open', 'high', 'low', 'close', 'volume' will be suffixed.
    current_cols = list(df.columns)
    # Identify features generated in this function call (not the original base OHLCV)
    generated_feature_cols = [col for col in current_cols if col not in base_cols]
    
    rename_dict = {col: col + suffix for col in generated_feature_cols}
    df.rename(columns=rename_dict, inplace=True)
    
    return df

# --- Time & Session Features (Keep as is) ---
def add_time_session_features(df):
    # ... (your existing add_time_session_features function) ...
    if not isinstance(df.index, pd.DatetimeIndex):
        print("Error: DataFrame index must be DatetimeIndex for time/session features.")
        return df
    df = df.copy()
    df['Hour_of_Day'] = df.index.hour
    df['Minute_of_Hour'] = df.index.minute
    df['Day_of_Week'] = df.index.dayofweek
    time_fraction_of_day = df['Hour_of_Day'] + df['Minute_of_Hour'] / 60.0
    df['Time_Sin'] = np.sin(2 * np.pi * time_fraction_of_day / 24.0)
    df['Time_Cos'] = np.cos(2 * np.pi * time_fraction_of_day / 24.0)
    df['Day_Sin'] = np.sin(2 * np.pi * df['Day_of_Week'] / 7.0)
    df['Day_Cos'] = np.cos(2 * np.pi * df['Day_of_Week'] / 7.0)
    df['Is_Asian_Session'] = ((df['Hour_of_Day'] >= 20) | (df['Hour_of_Day'] < 5)).astype(int)
    df['Is_London_Session'] = ((df['Hour_of_Day'] >= 3) & (df['Hour_of_Day'] < 12)).astype(int)
    df['Is_NY_Session'] = ((df['Hour_of_Day'] >= 8) & (df['Hour_of_Day'] < 17)).astype(int)
    df['Is_Overlap'] = ((df['Hour_of_Day'] >= 8) & (df['Hour_of_Day'] < 12)).astype(int)
    df['Is_US_Open_Hour'] = ((df['Hour_of_Day'] == 9) & (df['Minute_of_Hour'] >= 30) | (df['Hour_of_Day'] == 10) & (df['Minute_of_Hour'] < 30)).astype(int)
    df['Is_US_Close_Hour'] = ((df['Hour_of_Day'] == 15) | (df['Hour_of_Day'] == 16) & (df['Minute_of_Hour'] == 0)).astype(int)
    return df
