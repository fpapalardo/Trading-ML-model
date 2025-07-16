# data_handler.py

import os
import pandas as pd
from datetime import datetime, timedelta, timezone
from . import config
from . import binance_client

def get_latest_data():
    """
    Loads existing data, fetches all missing candles up to the last FULLY COMPLETED interval,
    and returns a combined DataFrame of closed candles only.
    """
    os.makedirs(config.DATA_FOLDER, exist_ok=True)
    
    df = pd.DataFrame()
    if os.path.exists(config.CANDLE_DATA_FILE):
        df = pd.read_parquet(config.CANDLE_DATA_FILE)
        
        # CRITICAL: Ensure timezone awareness
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')
    
    if not df.empty:
        last_timestamp = df.index[-1]  # Now already UTC aware
    else:
        print("No local data found. Fetching initial historical data...")
        last_timestamp = datetime.now(timezone.utc) - timedelta(days=60)

    # FIXED LOGIC: Get the most recent complete candle
    now_utc = datetime.now(timezone.utc)
    
    # Find the start of the current 5-minute interval
    current_interval_start = now_utc.replace(
        minute=(now_utc.minute // 5) * 5, 
        second=0, 
        microsecond=0
    )
    
    # The most recent complete candle starts one interval before the current one
    # We consider a candle complete if we're at least 0.2 seconds past its close time
    CANDLE_FINALIZATION_BUFFER = timedelta(seconds=0.2)
    
    # Calculate how far past the current interval start we are
    time_into_interval = now_utc - current_interval_start
    
    # Always use the previous interval as the safe end time
    # The candle that opened at (current_interval_start - 5min) is definitely complete
    # if we're past current_interval_start + buffer
    if time_into_interval >= CANDLE_FINALIZATION_BUFFER:
        # We're past the buffer, so the candle that just closed is complete
        safe_end_time = current_interval_start - timedelta(minutes=5)
    else:
        # We're at the exact boundary or just before the buffer
        # Be conservative and use the candle before the most recent one
        safe_end_time = current_interval_start - timedelta(minutes=10)
    
    print(f"Current time: {now_utc.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Most recent complete candle: {safe_end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Loop to fetch data in chunks until we are caught up to the safe end time
    while last_timestamp < safe_end_time:
        start_time = last_timestamp + timedelta(minutes=5)
        
        # For the API call, we need to fetch up to a time AFTER the candle we want
        # Add 1 minute to ensure we get the candle at safe_end_time
        api_end_time = safe_end_time + timedelta(minutes=1)
        
        start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
        end_time_str = api_end_time.strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"Fetching new candles from {start_time_str} up to {end_time_str}...")
        
        new_data = binance_client.client.get_historical_candles(
            config.TRADING_SYMBOL, 
            config.TIMEFRAME, 
            start_time_str, 
            end_str=end_time_str
        )
        
        if new_data.empty:
            print("No new candles returned. Data is up to date.")
            break
        
        # FIX: Ensure new_data has timezone-aware index before any operations
        if new_data.index.tz is None:
            new_data.index = pd.to_datetime(new_data.index).tz_localize('UTC')
        else:
            new_data.index = new_data.index.tz_convert('UTC')
        
        # VERIFY CANDLE COMPLETENESS
        # Filter to only include candles up to our safe end time
        new_data = new_data[new_data.index <= safe_end_time]
        
        if new_data.empty:
            print("No new complete candles to add.")
            break
        
        # Check that the last candle we're keeping is at or before our safe end time
        last_fetched_time = new_data.index[-1]
        
        if last_fetched_time > safe_end_time:
            print(f"ERROR: Filtered data still contains future candle at {last_fetched_time}")
            break
        
        # Verify we got the expected candle timestamps
        expected_timestamps = pd.date_range(
            start=start_time, 
            end=safe_end_time, 
            freq='5min',
            tz='UTC'  # Make expected timestamps timezone-aware
        )
        
        missing_candles = set(expected_timestamps) - set(new_data.index)
        if missing_candles:
            print(f"WARNING: Missing expected candles: {sorted(missing_candles)}")
        
        # Merge new data with existing data
        df = pd.concat([df, new_data])
        df = df[~df.index.duplicated(keep='last')]
        
        last_timestamp = df.index[-1]
        print(f"Caught up to {last_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

    if not df.empty:
        df.sort_index(inplace=True)
        # Save with explicit timezone info
        df.index.name = 'timestamp'
        df.to_parquet(config.CANDLE_DATA_FILE)
        print(f"Data updated and saved. Total candles: {len(df)}")
        print(f"Latest candle: {df.index[-1]} (Price: ${df.iloc[-1]['close']:.2f})")
        
    return df

def verify_data_integrity(df):
    """
    Verify that our data contains only complete candles and no gaps.
    """
    if df.empty:
        return True
    
    # Check for regular 5-minute intervals
    time_diffs = df.index.to_series().diff().dropna()
    expected_diff = pd.Timedelta(minutes=5)
    
    irregular_intervals = time_diffs[time_diffs != expected_diff]
    if not irregular_intervals.empty:
        print(f"WARNING: Found {len(irregular_intervals)} irregular time intervals in data")
        print(f"First few irregular intervals: {irregular_intervals.head()}")
        return False
    
    # Check that all timestamps are on 5-minute boundaries
    non_aligned = df.index[~((df.index.minute % 5 == 0) & (df.index.second == 0))]
    if len(non_aligned) > 0:
        print(f"WARNING: Found {len(non_aligned)} candles not aligned to 5-minute boundaries")
        print(f"First few non-aligned: {non_aligned[:5]}")
        return False
    
    print("Data integrity check passed - all candles appear to be complete")
    return True