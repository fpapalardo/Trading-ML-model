# common.py — shared constants and helpers for Optuna tuning
import os
from pathlib import Path
import optuna
from sklearn.model_selection import TimeSeriesSplit
from config import ROOT, DB_DIR, RAW_DIR

# —— Default configuration ——
DEFAULT_DB_DIR = DB_DIR
DEFAULT_SEED = 42
PRUNER_STARTUP_TRIALS = 5
CV_TRAIN_BLEND = (0.8, 0.2)  # (CV score weight, train score weight)
DEFAULT_MIN_TEST_SIZE = 1   # minimum test fold size when auto-calculating splits
DEFAULT_MAX_SPLITS = 4      # upper cap on splits if auto-calculating


def ensure_db_dir(db_dir: Path = DEFAULT_DB_DIR):
    """
    Make sure the Optuna database directory exists.
    """
    db_dir.mkdir(parents=True, exist_ok=True)


def get_storage_uri(study_name: str, db_dir: Path = DEFAULT_DB_DIR) -> str:
    """
    Construct a SQLite URI for the given study name.
    """
    ensure_db_dir(db_dir)
    filepath = db_dir / f"{study_name}.db"
    return f"sqlite:///{filepath}"


def create_study(
    study_name: str,
    direction: str = "minimize",
    db_dir: Path = DEFAULT_DB_DIR,
    seed: int = DEFAULT_SEED,
    pruner_startup: int = PRUNER_STARTUP_TRIALS
) -> optuna.study.Study:
    """
    Create or load an Optuna study with TimeSeriesSplit-friendly defaults.
    """
    storage = get_storage_uri(study_name, db_dir)
    return optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=pruner_startup),
        storage=storage,
        load_if_exists=True
    )


def ts_split(n_splits: int) -> TimeSeriesSplit:
    """
    Return a TimeSeriesSplit configured with the provided number of splits.
    """
    return TimeSeriesSplit(n_splits=n_splits)


def auto_ts_split(
    n_samples: int,
    min_test_size: int = DEFAULT_MIN_TEST_SIZE,
    max_splits: int = DEFAULT_MAX_SPLITS
) -> TimeSeriesSplit:
    """
    Automatically determine a reasonable n_splits for TS splitting based on sample count.

    Ensures each test fold has at least `min_test_size` samples and caps at `max_splits`.
    """
    # must have at least 2 splits
    if n_samples < (min_test_size * 3):
        n_splits = 2
    else:
        # n_splits+1 folds => approximate test_size = n_samples/(n_splits+1)
        possible = max(2, n_samples // min_test_size - 1)
        n_splits = min(possible, max_splits)

    print(f"{n_splits} splits are possible to use for optimization.\n")
    return TimeSeriesSplit(n_splits=n_splits)


def blend_scores(cv_score: float, train_score: float) -> float:
    """
    Blend CV and train scores using the configured weights in CV_TRAIN_BLEND.
    """
    w_cv, w_train = CV_TRAIN_BLEND
    return w_cv * cv_score + w_train * train_score
