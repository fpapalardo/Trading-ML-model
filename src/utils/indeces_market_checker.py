import pandas as pd

def load_databento_data(
    filepath: str,
    market_tz: str = 'America/New_York',
    start_date: str = '2019-01-01'
) -> pd.DataFrame:
    """
    Load a DataBento CSV containing multiple symbols,
    drop irrelevant columns, parse timestamps, convert to market timezone,
    filter from start_date onward, and format OHLC to 2 decimal floats.

    CSV must have columns:
      ts_event,rtype,publisher_id,instrument_id,
      open,high,low,close,volume,symbol

    Returns a DataFrame indexed by localized datetime with columns:
      open, high, low, close, volume, symbol
    """
    df = pd.read_csv(
        filepath,
        usecols=['ts_event','open','high','low','close','volume','symbol']
    )
    if df.empty:
        raise ValueError(f"No data found in file {filepath}")

    # Parse ts_event as UTC and convert to market timezone
    df['datetime'] = pd.to_datetime(df['ts_event'], utc=True)
    df['datetime'] = df['datetime'].dt.tz_convert(market_tz)
    df = df.set_index('datetime')

    # Filter from start_date onward
    start_ts = pd.to_datetime(start_date).tz_localize(market_tz)
    df = df[df.index >= start_ts]

    # Round OHLC to two decimals and ensure volume integer
    df[['open','high','low','close']] = (
        df[['open','high','low','close']]
        .astype(float)
        .round(2)
    )
    df['volume'] = df['volume'].astype(int)

    return df[['open','high','low','close','volume','symbol']]


def check_market_inconsistencies(
    df: pd.DataFrame
) -> None:
    """
    Check for any candles outside regular CME futures market hours using New York time:
      - Monday–Thursday after 17:01 ET
      - Friday after 17:01 ET
      - Any Saturday
      - Sunday before 18:00 ET
    Prints summary counts.
    """
    # Ensure index is in New York time
    df_ny = df.tz_convert('America/New_York')
    wd = df_ny.index.weekday
    hr = df_ny.index.hour
    minute = df_ny.index.minute

    # Define invalid conditions in Eastern Time
    mon_to_thu = (wd <= 3) & (hr == 17) & (minute >= 1)
    fri_after  = (wd == 4) & (((hr == 17) & (minute >= 1)) | (hr > 17))
    saturday   = (wd == 5)
    sunday     = (wd == 6) & (hr < 18)
    invalid = mon_to_thu | fri_after | saturday | sunday

    total = len(df_ny)
    count_invalid = invalid.sum()
    print(f"Total rows: {total}")
    print(f"Invalid rows: {count_invalid}")
    print(f"  Mon–Thu after 17:01 ET:  {mon_to_thu.sum()}")
    print(f"  Fri after 17:01 ET:     {fri_after.sum()}")
    print(f"  Saturday:               {saturday.sum()}")
    print(f"  Sun before 18:00 ET:    {sunday.sum()}")

    if count_invalid:
        print("Sample invalid timestamps in NY:")
        print(df_ny.index[invalid].strftime('%Y-%m-%d %H:%M').unique()[:10])

if __name__ == '__main__':
    # File containing all DataBento data
    filepath = 'databento_all.csv'
    # Load data from 2012-01-01 onward
    df = load_databento_data(filepath, start_date='2019-01-01')
    print(f"Loaded {len(df)} rows across {df['symbol'].nunique()} symbols since 2012-01-01.")

    # Check for market-hour inconsistencies
    check_market_inconsistencies(df)
