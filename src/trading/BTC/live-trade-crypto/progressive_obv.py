import pandas as pd
import numpy as np

class ProgressiveResamplingOBV:
    """
    Calculates OBV progressively for a specific timeframe by consuming
    base-timeframe candles. Used for live updates after initial priming.
    """
    def __init__(self, timeframe: str):
        self.timeframe = timeframe
        self.last_obv = 0.0
        self.last_close = 0.0
        self.buffer = pd.DataFrame()

    def update(self, new_candles_df: pd.DataFrame) -> pd.Series:
        if new_candles_df.empty:
            return pd.Series(dtype=np.float64)

        if self.buffer.empty:
            self.buffer = new_candles_df.copy()
        else:
            self.buffer = pd.concat([self.buffer, new_candles_df])

        self.buffer.index = pd.to_datetime(self.buffer.index)
        self.buffer = self.buffer[~self.buffer.index.duplicated(keep='last')]

        resampling_rules = {
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }
        completed_bars = self.buffer.resample(self.timeframe, label='right', closed='right').agg(resampling_rules).dropna()

        if completed_bars.empty:
            return pd.Series(dtype=np.float64)

        new_obv_values = {}
        for index, row in completed_bars.iterrows():
            if self.last_close != 0 and index <= getattr(self, '_last_processed_timestamp', pd.Timestamp(0)):
                continue

            new_obv = self.last_obv
            if self.last_close != 0:
                if row['close'] > self.last_close:
                    new_obv += row['volume']
                elif row['close'] < self.last_close:
                    new_obv -= row['volume']

            self.last_obv = new_obv
            self.last_close = row['close']
            new_obv_values[index] = new_obv
            self._last_processed_timestamp = index

        return pd.Series(new_obv_values, name=f"OBV_{self.timeframe}")