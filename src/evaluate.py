"""
Evaluate the trained failure model and write diagnostic figures.

Produces, under ``reports/figures/``:
  * confusion_matrix.png
  * roc_curve.png
  * precision_recall_curve.png
  * feature_importance.png  (permutation importance, model-agnostic)

    python src/evaluate.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless / no display
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "accidents.csv"
MODEL_PATH = ROOT / "models" / "failure_model.joblib"
FIG_DIR = ROOT / "reports" / "figures"

TARGET = "machine_failure"
DROP = [TARGET, "failure_type"]
RANDOM_STATE = 42


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit("No trained model found - run `python src/train.py` first.")

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=DROP)
    y = df[TARGET]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    model = joblib.load(MODEL_PATH)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Confusion matrix.
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    # ROC curve.
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_curve.png", dpi=120)
    plt.close(fig)

    # Precision-Recall curve (more informative under imbalance).
    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title("Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "precision_recall_curve.png", dpi=120)
    plt.close(fig)

    # Permutation feature importance (works for any fitted pipeline).
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, scoring="roc_auc"
    )
    order = result.importances_mean.argsort()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(X.columns[order], result.importances_mean[order], xerr=result.importances_std[order])
    ax.set_xlabel("Drop in ROC-AUC when shuffled")
    ax.set_title("Permutation Feature Importance")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_importance.png", dpi=120)
    plt.close(fig)

    print(f"Saved 4 figures to {FIG_DIR}")
    ranked = sorted(
        zip(X.columns, result.importances_mean), key=lambda t: t[1], reverse=True
    )
    print("\nTop predictive features (permutation importance):")
    for name, imp in ranked[:5]:
        print(f"  {name:>22s}  {imp:.4f}")


if __name__ == "__main__":
    main()
