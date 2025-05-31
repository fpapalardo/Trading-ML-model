import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from datetime import timedelta

def check_overfit(model, X_tr, X_te, y_tr, y_te):
    train_preds = model.predict(X_tr)
    test_preds = model.predict(X_te)
    train_mse = mean_squared_error(y_tr, train_preds)
    test_mse = mean_squared_error(y_te, test_preds)
    ratio = test_mse / train_mse if train_mse != 0 else float('inf')

    print(f"\n📉 Overfitting check:")
    print(f"Train MSE: {train_mse:.8f}")
    print(f"Test MSE:  {test_mse:.8f}")
    print(f"Overfit ratio (Test / Train): {ratio:.2f}")

    if ratio > 2.0:
        print("🚨 Overfitting: Model performs poorly on unseen data.")
    elif ratio > 1.2:
        print("⚠️ Mild overfitting: Model may be too complex.")
    elif ratio < 0.8:
        print("⚠️ Possible underfitting: Model may be too simple.")
    else:
        print("✅ Good generalization between train and test.")

def generate_oof_predictions(models, X, y, splits):
    """
    Generates out-of-fold predictions for a list of models using TimeSeriesSplit.

    Parameters:
    - models: list of sklearn-style models (will be cloned per fold)
    - X: feature DataFrame
    - y: target Series
    - n_splits: number of TSCV splits

    Returns:
    - oof_df: DataFrame of shape (len(X), len(models)) with OOF predictions
    """
    oof_preds = np.zeros((len(X), len(models)))
    tscv = TimeSeriesSplit(n_splits=splits)

    for i, model in enumerate(models):
        for train_idx, val_idx in tscv.split(X):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val = X.iloc[val_idx]

            fold_model = clone(model)
            fold_model.fit(X_train, y_train)
            oof_preds[val_idx, i] = fold_model.predict(X_val)

    return pd.DataFrame(oof_preds, index=X.index, columns=[f'model_{i}' for i in range(len(models))])

def session_key(ts: pd.Timestamp) -> pd.Timestamp:
    # shift back 18 h, then floor to midnight to get a unique session “date”
    return (ts - timedelta(hours=18)).normalize()

def is_same_session(start_time: pd.Timestamp, end_time: pd.Timestamp) -> bool:
    return session_key(start_time) == session_key(end_time)

def avoid_news(row):
    ts = row["datetime"]
    return any(start <= ts <= end for (start, end) in news_windows)

def avoid_hour_18_19(row):
    """
    Avoid trading in the first hour of the session (18:00 to 19:00 inclusive).
    """
    if not pd.api.types.is_datetime64_any_dtype(row['datetime']):
        return False
    hour = row['datetime'].hour
    return hour == 18

