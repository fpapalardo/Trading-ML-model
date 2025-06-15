import re
import pandas as pd

EMA_PATTERN = re.compile(r'^EMA_\d+')

def price_vs_all_mas(
    df: pd.DataFrame,
    price_col: str = 'close'
) -> pd.DataFrame:
    """
    For each column matching EMA_<number>, add price_col_vs_EMA_<number> = price / EMA.
    """
    for ema in filter(EMA_PATTERN.match, df.columns):
        df[f"{price_col}_vs_{ema}"] = (
            pd.to_numeric(df[price_col], errors='coerce')
            / pd.to_numeric(df[ema],        errors='coerce')
        )
    return df


def ma_vs_all_mas(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    For every pair of EMAs in the DataFrame, add EMAi_vs_EMAj = EMAi / EMAj.
    Only i<j to avoid duplicates; gives you all combinations.
    """
    emas = [c for c in df.columns if EMA_PATTERN.match(c)]
    for i, ema_i in enumerate(emas):
        for ema_j in emas[i+1:]:
            df[f"{ema_i}_vs_{ema_j}"] = (
                pd.to_numeric(df[ema_i], errors='coerce')
                / pd.to_numeric(df[ema_j], errors='coerce')
            )
    return df


def ma_slope_all_mas(
    df: pd.DataFrame,
    periods: int = 1
) -> pd.DataFrame:
    """
    For each EMA_<n> column, add EMA_<n>_slope_<periods> = diff(periods)/periods.
    """
    for ema in filter(EMA_PATTERN.match, df.columns):
        df[f"{ema}_slope_{periods}"] = (
            pd.to_numeric(df[ema], errors='coerce')
            .diff(periods)
            / periods
        )
    return df

def lagged_features(df: pd.DataFrame,
                    cols: list[str] | None = None,
                    lags: tuple[int, ...] = (1, 3, 6)
) -> pd.DataFrame:
    """
    If cols is None, we take *all* columns in df.
    """
    if cols is None:
        cols = df.columns.tolist()

    for c in cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors='coerce')
            for lag in lags:
                df[f"{c}_lag_{lag}"] = s.shift(lag)
    return df