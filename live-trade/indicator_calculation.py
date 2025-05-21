import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from ta.trend import MACD


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # === Candle range
    def choppiness_index(high, low, close, length=14):
        tr = AverageTrueRange(high=high, low=low, close=close, window=length).average_true_range()
        atr_sum = tr.rolling(length).sum()
        high_max = high.rolling(length).max()
        low_min = low.rolling(length).min()
        return 100 * np.log10(atr_sum / (high_max - low_min)) / np.log10(length)

    def detect_pivot_highs_lows(df, lookback, lookforward):
        highs, lows = df['high'], df['low']
        df[f'is_pivot_high_{lookback}'] = (
            (highs.shift(lookback)  < highs) &
            (highs.shift(-lookforward)< highs)
        ).astype(int)
        df[f'is_pivot_low_{lookback}']  = (
            (lows.shift(lookback)   > lows) &
            (lows.shift(-lookforward)> lows)
        ).astype(int)
        return df

    # === Feature Engineering ===
    df['atr_5']         = AverageTrueRange(df['high'], df['low'], df['close'], window=5).average_true_range()
    df['atr_pct']       = df['atr_5'] / df['close']
    df['rsi_6']         = RSIIndicator(df['close'], window=6).rsi()
    macd = MACD(df['close'], window_fast=6, window_slow=13, window_sign=5)
    df['macd_fast']     = macd.macd()
    df['macd_fast_diff']= macd.macd_diff()
    df['chop_index'] = choppiness_index(df['high'], df['low'], df['close'])
    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_dist'] = (df['close'] - df['ema_9'])
    
    df = detect_pivot_highs_lows(df, 5, 5)
    df = detect_pivot_highs_lows(df,10,10)
    df = detect_pivot_highs_lows(df,15,15)

    df['hour'] = df['datetime'].dt.hour

    return df
