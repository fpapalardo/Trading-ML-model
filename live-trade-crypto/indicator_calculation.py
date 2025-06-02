import pandas as pd
import numpy as np
from ta.volatility import AverageTrueRange
from ta.momentum    import RSIIndicator
from ta.trend       import MACD, EMAIndicator, ADXIndicator
from ta.volatility  import AverageTrueRange, BollingerBands


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # === Candle range
    def choppiness_index(high, low, close, length=14):
        tr = AverageTrueRange(high=high, low=low, close=close, window=length).average_true_range().shift(1)
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
    
    def assign_session_hourly(ts):
        hour = ts.hour + ts.minute / 60.0
        if 18 <= hour or hour < 2:      # 18:00 - 02:00
            return "asia"
        elif 2 <= hour < 4:             # 02:00 - 04:00
            return "london_pre"
        elif 4 <= hour < 8:             # 04:00 - 08:00
            return "london"
        elif 8 <= hour < 9.5:           # 08:00 - 09:30
            return "ny_pre"
        elif 9.5 <= hour < 12:          # 09:30 - 12:00
            return "ny"
        elif 12 <= hour < 15:          # 12:00 - 15:00
            return "ny_pm"
        elif 15 <= hour < 17:          # 15:00 - 17:00
            return "ny_close"
        else:                          # 17:00 - 18:00
            return "overnight"

    # === Feature Engineering ===
    df['atr_5']         = AverageTrueRange(df['high'], df['low'], df['close'], window=5).average_true_range().shift(1)
    df['atr_pct']       = df['atr_5'] / df['close']
    df['rsi_6']         = RSIIndicator(df['close'], window=6).rsi().shift(1)
    macd = MACD(df['close'], window_fast=6, window_slow=13, window_sign=5)
    df['macd_fast']     = macd.macd().shift(1)
    df['macd_fast_diff']= macd.macd_diff().shift(1)
    df['chop_index'] = choppiness_index(df['high'], df['low'], df['close']).shift(1)
    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean().shift(1)
    df['ema_dist'] = (df['close'] - df['ema_9']).shift(1)

    df['volume'] = df['volume'].astype(float)
    df['volume_delta'] = df['volume'].diff()
    df['volume_delta_ema'] = df['volume_delta'].ewm(span=10, adjust=False).mean().shift(1)

    # === Bollinger Band metrics ===
    bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bollinger_width'] = (bb.bollinger_hband() - bb.bollinger_lband()).shift(1)
    df['bollinger_z'] = ((df['close'] - bb.bollinger_mavg()) / (bb.bollinger_hband() - bb.bollinger_lband())).shift(1)

    # === ADX (trend strength) ===
    df['adx'] = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14).adx().shift(1)

    # === EMA slope ===
    ema = EMAIndicator(close=df['close'], window=21)
    df['ema_21'] = ema.ema_indicator()
    df['ema_slope'] = df['ema_21'].diff().shift(1)

    session_mapping = {
        'asia': 0,
        'london_pre': 1,
        'london': 2,
        'ny_pre': 3,
        'ny': 4,
        'ny_pm': 5,
        'ny_close': 6
    }
    df['session_code'] = df['datetime'].apply(assign_session_hourly).map(session_mapping)
    
    df = detect_pivot_highs_lows(df, 5, 5)
    df = detect_pivot_highs_lows(df,10,10)
    df = detect_pivot_highs_lows(df,15,15)

    df['hour'] = df['datetime'].dt.hour

    return df
