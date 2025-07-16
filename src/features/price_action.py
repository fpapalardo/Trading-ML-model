import pandas as pd
import numpy as np

# --- Price Action & Patterns ---
def candle_features(df):
    """Append wick, body, and related metrics."""
    h, l, o, c = df['high'], df['low'], df['open'], df['close']
    df['Candle_Range'] = h-l
    df['Candle_Body'] = (c-o).abs()
    df['Upper_Wick'] = h-np.maximum(o,c)
    df['Lower_Wick'] = np.minimum(o,c)-l
    return df

def return_features(df):
    """Append log and simple returns."""
    p = pd.to_numeric(df['close'], errors='coerce').replace(0,np.nan)
    df['Log_Ret_1'] = np.log(p/p.shift(1))
    df['Simple_Ret_1'] = p.pct_change()
    return df

def prev_swing_high_low(df, window=12):
    df['Prev_Sw_High'] = df['high'].rolling(window).max().shift(1)
    df['Prev_Sw_Low']  = df['low'].rolling(window).min().shift(1)
    return df

def dist_to_closest_sr(df):
    df['Dist_High'] = (df['Prev_Sw_High']-df['close']).abs()
    df['Dist_Low']  = (df['Prev_Sw_Low']-df['close']).abs()
    df['Dist_SR']   = df[['Dist_High','Dist_Low']].min(axis=1)
    return df

import pandas as pd

def add_prev_swing_dist(
    df: pd.DataFrame,
    window: int = 12,
    atr_col: str = "ATR_14"
) -> pd.DataFrame:
    """
    Ensure Prev_Swing_Dist exists:
      - Prev_Sw_High / Prev_Sw_Low by rolling window if missing
      - Prev_Swing_Dist = (close - midpoint) / ATR_14 (or raw if no ATR_14)
    """
    df = df.copy()

    # 1) If they've already got it, nothing to do
    if "Prev_Swing_Dist" in df.columns:
        return df

    # 2) Ensure Prev_Sw_High & Prev_Sw_Low
    if "Prev_Sw_High" not in df.columns or "Prev_Sw_Low" not in df.columns:
        # rolling max/min then shift
        df["Prev_Sw_High"] = df["high"].rolling(window=window).max().shift(1)
        df["Prev_Sw_Low"]  = df["low"].rolling(window=window).min().shift(1)

    # 3) midpoint between last swing high & low
    swing_mid = 0.5 * (df["Prev_Sw_High"] + df["Prev_Sw_Low"])

    # 4) normalized by ATR if available
    if atr_col in df.columns:
        df["Prev_Swing_Dist"] = (df["close"] - swing_mid) / df[atr_col]
    else:
        df["Prev_Swing_Dist"] = df["close"] - swing_mid

    return df


def candlestick_patterns(df):
    df['Bull_Engulf'] = ((df['close']>df['open'])&(df['open']<df['close'].shift(1))&(df['close']>df['open'].shift(1))).astype(int)
    df['Bear_Engulf'] = ((df['close']<df['open'])&(df['open']>df['close'].shift(1))&(df['close']<df['open'].shift(1))).astype(int)
    return df

def stop_hunt(df):
    upper = df['high']-np.maximum(df['open'],df['close'])
    lower = np.minimum(df['open'],df['close'])-df['low']
    rng = df['high']-df['low']+1e-6
    df['Stop_Hunt'] = ((upper>0.6*rng)|(lower>0.6*rng)).astype(int)
    return df

def fvg(df):
    """
    Calculates Fair Value Gaps (FVG) without lookahead bias.

    A Fair Value Gap is a three-candle pattern. This function identifies the
    pattern and makes the signal available for the candle immediately following
    the completion of the pattern, ensuring no future data is used.

    The pattern is defined by three consecutive candles:
    - Candle 1 (T-2): Two periods ago
    - Candle 2 (T-1): One period ago
    - Candle 3 (T): The current candle being evaluated

    A Bullish FVG is formed if the low of Candle 3 is higher than the high of Candle 1.
    A Bearish FVG is formed if the high of Candle 3 is lower than the low of Candle 1.
    """
    df = df.copy()

    # --- Get Past Data Using Shifts ---
    # To evaluate the pattern at the current candle (T), we need data from T-2.
    high_t2 = df['high'].shift(2)
    low_t2 = df['low'].shift(2)

    # --- Identify the Gaps ---
    # The pattern is confirmed at the close of the current candle (T).
    # A bullish FVG exists if the current low is above the high from 2 periods ago.
    bullish_fvg = df['low'] > high_t2

    # A bearish FVG exists if the current high is below the low from 2 periods ago.
    bearish_fvg = df['high'] < low_t2

    # --- Calculate Features ---
    # The signal is known at the close of candle T, so it is valid for making
    # a decision for the *next* candle, T+1. We shift all final results by 1.

    # 1. FVG Existence: 1 for bullish, -1 for bearish, 0 for none.
    fvg_exists = pd.Series(0, index=df.index)
    fvg_exists.loc[bullish_fvg] = 1
    fvg_exists.loc[bearish_fvg] = -1
    df['FVG_Exists'] = fvg_exists.shift(1).fillna(0).astype(int)

    # 2. FVG Size: The magnitude of the gap in price points.
    fvg_size = pd.Series(0.0, index=df.index)
    fvg_size.loc[bullish_fvg] = df['low'][bullish_fvg] - high_t2[bullish_fvg]
    fvg_size.loc[bearish_fvg] = low_t2[bearish_fvg] - df['high'][bearish_fvg]
    df['FVG_Size'] = fvg_size.shift(1).fillna(0)

    # 3. FVG Position: Where the close of the 3rd candle is relative to the gap it created.
    #    - For a bullish gap, a value of 1 means the close was at the top of the gap.
    #    - For a bearish gap, a value of 1 means the close was at the bottom of the gap.
    #    - A value of 0.5 means the close was in the middle of the gap.
    fvg_pos = pd.Series(np.nan, index=df.index)
    
    # Calculate normalized position for bullish gaps
    bull_gap_size = df['low'][bullish_fvg] - high_t2[bullish_fvg]
    bull_filled = df['close'][bullish_fvg] - high_t2[bullish_fvg]
    fvg_pos.loc[bullish_fvg] = (bull_filled / bull_gap_size).clip(0, 1) # Clip to handle closes outside the gap

    # Calculate normalized position for bearish gaps
    bear_gap_size = low_t2[bearish_fvg] - df['high'][bearish_fvg]
    bear_filled = low_t2[bearish_fvg] - df['close'][bearish_fvg]
    fvg_pos.loc[bearish_fvg] = (bear_filled / bear_gap_size).clip(0, 1) # Clip to handle closes outside the gap
    
    df['FVG_Pos'] = fvg_pos.shift(1).fillna(0)

    return df

def day_high_low_open(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates point-in-time daily high, low, and open prices
    without lookahead bias.
    """
    df = df.copy()

    # Group by the calendar date of the DatetimeIndex
    daily_groups = df.groupby(df.index.date)

    # High_Day: The max high seen *so far* today
    df['High_Day'] = daily_groups['high'].expanding().max().droplevel(0)

    # Low_Day: The min low seen *so far* today
    df['Low_Day'] = daily_groups['low'].expanding().min().droplevel(0)

    # Open_Day: The very first 'open' price of the day
    df['Open_Day'] = daily_groups['open'].transform('first')

    return df

def prev_high_low(df):
    df['Prev_High']=df['high'].shift(1)
    df['Prev_Low']=df['low'].shift(1)
    return df

def price_vs_open(df):
    df['Price_vs_Open'] = df['close'] - df['Open_Day']
    return df