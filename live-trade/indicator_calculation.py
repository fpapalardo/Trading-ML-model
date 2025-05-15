import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from ta.trend import MACD


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # === Candle range
    df['candle_range'] = df['high'] - df['low']

    # === RSI (6)
    df['rsi_6'] = RSIIndicator(close=df['close'], window=6).rsi()

    # === ATR (5)
    df['atr_5'] = AverageTrueRange(
        high=df['high'], low=df['low'], close=df['close'], window=5
    ).average_true_range()

    # === ATR percentage of close
    df['atr_14'] = AverageTrueRange(
        high=df['high'], low=df['low'], close=df['close'], window=14
    ).average_true_range()
    df['atr_pct'] = df['atr_14'] / df['close']

    # === MACD and MACD difference
    macd = MACD(close=df['close'])
    df['macd_fast'] = macd.macd()
    df['macd_fast_diff'] = macd.macd_diff()

    # === 1-bar return
    df['return_1'] = df['close'].pct_change(periods=1)

    # === Pivot logic
    def pivot_high(series, left=2, right=2):
        return series[(series.shift(left) < series) & (series.shift(-right) < series)]

    def pivot_low(series, left=2, right=2):
        return series[(series.shift(left) > series) & (series.shift(-right) > series)]

    df['is_pivot_high_5'] = 0
    df['is_pivot_low_5'] = 0
    df.loc[pivot_high(df['high'], left=2, right=2).index, 'is_pivot_high_5'] = 1
    df.loc[pivot_low(df['low'], left=2, right=2).index, 'is_pivot_low_5'] = 1

    df['is_pivot_high_10'] = 0
    df['is_pivot_low_10'] = 0
    df.loc[pivot_high(df['high'], left=5, right=5).index, 'is_pivot_high_10'] = 1
    df.loc[pivot_low(df['low'], left=5, right=5).index, 'is_pivot_low_10'] = 1

    df.fillna(-999, inplace=True)
    return df
