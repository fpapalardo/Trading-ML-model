import pandas as pd

# Load data
df = pd.read_csv(
    'data/NQ/databento_all.csv',
    parse_dates=['ts_event'],
    usecols=['symbol','ts_event','open','high','low','close','volume']
)
df.rename(columns={'ts_event': 'datetime'}, inplace=True)

# Known quarterly NQ roll dates (you can update this dict as new contracts appear)
ROLL_DATES = [
    # Format: (contract, roll datetime in NY time)
    ("NQH0", "2020-03-19 17:00:00-04:00"),
    ("NQM0", "2020-06-18 17:00:00-04:00"),
    ("NQU0", "2020-09-17 17:00:00-04:00"),
    ("NQZ0", "2020-12-17 17:00:00-05:00"),
    ("NQH1", "2021-03-18 17:00:00-04:00"),
    ("NQM1", "2021-06-17 17:00:00-04:00"),
    ("NQU1", "2021-09-16 17:00:00-04:00"),
    ("NQZ1", "2021-12-16 17:00:00-05:00"),
    ("NQH2", "2022-03-17 17:00:00-04:00"),
    ("NQM2", "2022-06-16 17:00:00-04:00"),
    ("NQU2", "2022-09-15 17:00:00-04:00"),
    ("NQZ2", "2022-12-15 17:00:00-05:00"),
    ("NQH3", "2023-03-16 17:00:00-04:00"),
    ("NQM3", "2023-06-15 17:00:00-04:00"),
    ("NQU3", "2023-09-21 17:00:00-04:00"),
    ("NQZ3", "2023-12-21 17:00:00-05:00"),
    ("NQH4", "2024-03-21 17:00:00-04:00"),
    ("NQM4", "2024-06-20 17:00:00-04:00"),
    ("NQU4", "2024-09-19 17:00:00-04:00"),
    ("NQZ4", "2024-12-19 17:00:00-05:00"),
    ("NQH5", "2025-03-20 17:00:00-04:00"),
    ("NQM5", "2025-06-19 17:00:00-04:00"),
    ("NQU5", "2025-09-18 17:00:00-04:00"),
    ("NQZ5", "2025-12-18 17:00:00-05:00"),
]

# Build windows for each contract
windows = []
for i, (sym, roll_dt) in enumerate(ROLL_DATES):
    roll = pd.Timestamp(roll_dt).tz_convert('America/New_York')
    if i == 0:
        start = df['datetime'].min().tz_localize('America/New_York') if df['datetime'].dt.tz is None else df['datetime'].min()
    else:
        start = pd.Timestamp(ROLL_DATES[i-1][1]).tz_convert('America/New_York')
    windows.append((sym, start, roll))

# Select the best contract for each window
parts = []
for sym, start, end in windows:
    mask = (
        (df['symbol'] == sym)
        & (df['datetime'] >= start)
        & (df['datetime'] < end)
    )
    part = df.loc[mask].copy()
    parts.append(part)

df_roll = pd.concat(parts).sort_values('datetime')

# Final touch: drop any negative OHLC
df_roll = df_roll[(df_roll[['open', 'high', 'low', 'close']] >= 0).all(axis=1)]

# Save result
df_roll.to_csv('data/NQ/databento_continuous_clean.csv', index=False)
print("Done. Continuous, clean, roll-adjusted file written.")

