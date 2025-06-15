import pandas as pd

# 1) Load both Parquets
df_training = pd.read_parquet("labeled_data_6NQ.parquet")
df_live     = pd.read_parquet("live.parquet")

# 2) TZ-align the live index to the training index
train_tz = df_training.index.tz
if train_tz is not None and df_live.index.tz is None:
    df_live.index = pd.to_datetime(df_live.index).tz_localize(train_tz)

# 3) Filter both to just May 5, 2025
start = pd.to_datetime("2025-06-10").tz_localize(train_tz)
end   = start + pd.Timedelta(days=1)
train_day = df_training.loc[start:end, ["open","high","low","close","volume"]]

# 4) Align on timestamps and compare
common_idx = train_day.index.intersection(train_day.index)
train_bars = train_day.loc[common_idx]

# 5) Find and print any mismatches
import pandas as pd

# assume train_bars and live_bars are your aligned DataFrames with OHLCV for May 5

cols = ["open","high","low","close","volume"]

# 1) Find where they differ

print(train_bars[["open","high","low","close","volume"]])