from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

def evaluate_model(name, model, Xtr, Xte, ytr, yte):
    train_preds = model.predict(Xtr)
    test_preds = model.predict(Xte)

    train_acc = accuracy_score(ytr, train_preds)
    test_acc = accuracy_score(yte, test_preds)

    print(f"\n📊 {name} Classification Accuracy:")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    return test_preds

def classification_insights(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compute and print a suite of classification metrics and distributions.

    Prints:
      - Train / test target distribution
      - Test prediction distribution
      - Accuracy & macro-F1
      - sklearn.classification_report
      - Confusion matrix
      - (If binary) ROC-AUC

    Returns a dict with:
      - 'y_pred':   np.ndarray
      - 'accuracy': float
      - 'f1_macro': float
      - 'confusion_matrix': np.ndarray
      - 'classification_report': str
      - 'roc_auc': Optional[float]
    """
    # 1) raw predictions
    y_pred = model.predict(X_test)

    # 2) target distributions
    print("=== Target distribution (train) ===")
    print(y_train.value_counts(normalize=True).rename("proportion"))
    print("\n=== Target distribution (test) ===")
    print(y_test.value_counts(normalize=True).rename("proportion"))

    # 3) prediction distribution
    print("\n=== Prediction distribution (test) ===")
    print(pd.Series(y_pred, name="pred").value_counts(normalize=True))

    # 4) accuracy & macro-F1
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"\nAccuracy: {acc:.4f}")
    print(f"F1 Score (macro): {f1:.4f}")

    # 5) classification report
    print("\nClassification report:")
    rpt = classification_report(
        y_test, y_pred, target_names=class_names, zero_division=0
    )
    print(rpt)

    # 6) confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:")
    print(cm)

    # 7) optional ROC-AUC for binary
    roc_auc: Optional[float] = None
    if hasattr(model, "predict_proba") and len(np.unique(y_test)) == 2:
        proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, proba)
        print(f"\nROC AUC: {roc_auc:.4f}")

    # pack results
    return {
        "y_pred": y_pred,
        "accuracy": acc,
        "f1_macro": f1,
        "classification_report": rpt,
        "confusion_matrix": cm,
        "roc_auc": roc_auc,
    }

def compute_max_consecutive_loss(df):
    """
    Returns:
      max_loss:  float, worst drawdown from any run of trades,
                 where run_sum resets to 0 on any net gain.
      start:     Timestamp of the first losing trade in that run
      end:       Timestamp of the last losing trade in that run
    """
    pnl = df['pnl'].values
    times = df['entry_time'].values

    run_sum = 0.0
    max_loss = 0.0
    run_start_idx = 0

    best_start_idx = 0
    best_end_idx   = 0

    for i, x in enumerate(pnl):
        run_sum += x

        # If we've bounced back to >= 0, start a fresh run at next trade
        if run_sum >= 0:
            run_sum = 0.0
            run_start_idx = i + 1
            continue

        # Otherwise, we're in a drawdown; record its depth
        if -run_sum > max_loss:
            max_loss      = -run_sum
            best_start_idx = run_start_idx
            best_end_idx   = i

    return (
        max_loss,
        pd.to_datetime(times[best_start_idx]),
        pd.to_datetime(times[best_end_idx])
    )


def visualize_results(results):
    best_result = None
    for r in results:
        df = r['results'].copy()
        df = df.sort_values(by='entry_time')
        df['cumulative_pnl'] = df['pnl'].cumsum()

        # Count how many trades exited for each reason
        exit_counts = df['exit_reason'].value_counts(dropna=False)
        print(exit_counts)

        if (
            df['cumulative_pnl'].iloc[-1] > 0 and
            r['sharpe'] > 0.01 and
            r['trades'] > 1 and
            r['win_rate'] > 0.001 and
            r['profit_factor'] > 0.01 and
            r['expectancy'] > 0.01 and
            r['pnl'] > 1
        ):
            if best_result is None or r['sharpe'] > best_result['sharpe']:
                best_result = r.copy()
                best_result['cumulative_pnl'] = df['cumulative_pnl']
                best_result['entry_time'] = df['entry_time']

                # === Calculate max drawdown (largest PnL loss from peak)
                cumulative = df['cumulative_pnl']
                rolling_max = cumulative.cummax()
                drawdowns = cumulative - rolling_max
                max_drawdown = drawdowns.min()  # Most negative drop
                max_drawdown_start = rolling_max[drawdowns.idxmin()]
                best_result['max_drawdown'] = max_drawdown

    # === Plot the best one ===
    # === After determining best_result
    if best_result:
        df = best_result['results'].copy()
        df = df.sort_values(by='entry_time')
        df['cumulative_pnl'] = df['pnl'].cumsum()

        max_loss, loss_start, loss_end = compute_max_consecutive_loss(df)

        # === Plot
        plt.figure(figsize=(12, 4))
        plt.plot(df['entry_time'], df['cumulative_pnl'], label='Cumulative PnL', color='green')
        plt.axvspan(loss_start, loss_end, color='red', alpha=0.2, label='Max Loss Window')
        plt.title(f"Top Sharpe Strategy | Max Consecutive Loss: {max_loss:.2f} | Cumulative PnL: {best_result['pnl']:.2f}")
        plt.xlabel("Datetime")
        plt.ylabel("Cumulative PnL")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        print(f"💣 Max Consecutive PnL Loss: {max_loss:.2f}")
        print(f"📆 Period: {loss_start} → {loss_end}")
        best_result['results'].to_csv("best_strategy_results.csv", index=False)
        print("✅ Saved best_strategy_results.csv")
    else:
        print("❌ No strategy met the conditions.")