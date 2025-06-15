import pandas as pd
from pathlib import Path
from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")

def load_local_csv(csv_path: Path, tz: ZoneInfo = NY_TZ) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        parse_dates=['datetime'],
        index_col='datetime'
    )
    # Clean up column names (strip whitespace/quotes)
    df.columns = (
        df.columns
          .str.strip()
          .str.replace(r"^['\"]|['\"]$", "", regex=True)
    )
    # Ensure required OHLCV columns
    required = {'open','high','low','close','volume'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path.name}: {missing}")
    # Localize/convert timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize(tz)
    else:
        df.index = df.index.tz_convert(tz)
    return df[['open','high','low','close','volume']]


def resample_bars(
    df: pd.DataFrame,
    rule: str,
    closed: str = 'right',
    label: str  = 'right'
) -> pd.DataFrame:
    """
    Generic OHLCV resampler using a dict for .agg() to avoid the 'func' error.
    """
    agg_dict = {
        'open':   'first',
        'high':   'max',
        'low':    'min',
        'close':  'last',
        'volume': 'sum'
    }
    return (
        df
        .resample(rule, closed=closed, label=label)
        .agg(agg_dict)
        .dropna()
    )


def load_and_resample_data(
    market:     str,
    timeframes: list[str],
    csv_dir:    Path,
    tz:         ZoneInfo = NY_TZ
) -> tuple[pd.DataFrame, dict[str,pd.DataFrame]]:
    """
    1) Finds CSVs in csv_dir (filters by `market` if multiple)
    2) Loads & concatenates them into one tz-aware DataFrame
    3) Resamples into each requested timeframe
    """
    csv_dir = Path(csv_dir)
    all_csvs = list(csv_dir.glob("*.csv"))
    if not all_csvs:
        raise FileNotFoundError(f"No CSVs found in {csv_dir}")

    # If multiple, pick those whose filename contains the market name
    if len(all_csvs) > 1:
        matched = [f for f in all_csvs if market.lower() in f.name.lower()]
        if matched:
            csvs_to_load = matched
        else:
            raise RuntimeError(
                f"Multiple CSVs found but none match '{market}':\n" +
                "\n".join(f.name for f in all_csvs)
            )
    else:
        csvs_to_load = all_csvs

    # Load & concatenate
    pieces = [load_local_csv(p, tz) for p in csvs_to_load]
    df_raw = pd.concat(pieces).sort_index()

    # Resample each timeframe
    resampled = {
        tf: resample_bars(df_raw, rule=tf)
        for tf in timeframes
    }

    return df_raw, resampled
