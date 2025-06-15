import pandas as pd
import os

# === Config ===
INPUT_FILE = 'ohlcv.csv'  # update path as needed
OUTPUT_FILE = 'cleaned_data.csv'  # or use .parquet
IS_PARQUET = OUTPUT_FILE.endswith('.parquet')

# === Step 1: Load raw file ===
df = pd.read_csv(INPUT_FILE, sep=';', header=None,
                 names=['datetime', 'open', 'high', 'low', 'close', 'volume'])

# === Step 2: Parse datetime and localize to UTC (no conversion)
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df['datetime'] = df['datetime'].dt.tz_localize('UTC')

# === Step 3: Drop invalid rows, sort by time
df.dropna(subset=['datetime'], inplace=True)
df.sort_values('datetime', inplace=True)

# === Step 4: Save cleaned output ===
if IS_PARQUET:
    df.to_parquet(OUTPUT_FILE, index=False)
else:
    df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Cleaned file saved to: {os.path.abspath(OUTPUT_FILE)}")
