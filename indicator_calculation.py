import pandas as pd
import numpy as np
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # === EMA Cross ===
    df['ema_9'] = EMAIndicator(close=df['close'], window=9).ema_indicator()
    df['ema_21'] = EMAIndicator(close=df['close'], window=21).ema_indicator()

    # === MACD ===
    df['macd'] = MACD(close=df['close']).macd_diff()

    # === ATR ===
    df['atr_14'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()

    # === RSI ===
    df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()

    # === VWAP Difference ===
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    df['vwap_diff'] = df['close'] - df['vwap']

    # === Candle Body Metrics ===
    df['body_pct'] = (df['close'] - df['open']).abs() / (df['high'] - df['low'] + 1e-9)
    df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']

    # === Volume Delta EMA ===
    df['volume_delta'] = df['volume'].diff().fillna(0)
    df['volume_delta_ema'] = df['volume_delta'].ewm(span=10).mean()

    # === Choppiness Index ===
    high_low_diff = df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min()
    atr_sum = df['atr_14'].rolling(window=14).sum()
    df['chop_index'] = 100 * np.log10(atr_sum / high_low_diff) / np.log10(14)

    # Drop rows with NaNs
    df.dropna(inplace=True)
    return df
