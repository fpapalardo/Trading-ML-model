from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
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