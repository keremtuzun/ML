"""
Train and select a predictive model for mechanical failure.

Pipeline:
  1. Load the (synthetic) accident dataset, generating it if missing.
  2. Build a preprocessing + model pipeline (scaling numeric features,
     one-hot encoding the categorical quality level).
  3. Compare several classifiers with stratified cross-validation, scoring
     by ROC-AUC (robust to the heavy class imbalance of failure data).
  4. Refit the best model on the full training split and persist it,
     along with a JSON metrics summary, to ``models/``.

    python src/train.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "accidents.csv"
MODEL_DIR = ROOT / "models"

TARGET = "machine_failure"
CATEGORICAL = ["machine_quality"]
# Everything else numeric and predictive (drop the leaky failure_type label).
DROP = [TARGET, "failure_type"]

RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    """Load the dataset, generating it on first run."""
    if not DATA_PATH.exists():
        print("No dataset found - generating one with src/generate_data.py defaults.")
        from generate_data import generate

        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        generate().to_csv(DATA_PATH, index=False)
    return pd.read_csv(DATA_PATH)


def build_preprocessor(numeric: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )


def candidate_models() -> dict[str, object]:
    """Classifiers to compare. class_weight handles the imbalance."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def main() -> None:
    df = load_data()
    numeric = [c for c in df.columns if c not in DROP + CATEGORICAL]
    X = df.drop(columns=DROP)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor(numeric)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print(f"Training on {len(X_train):,} rows, holding out {len(X_test):,} for test.")
    print(f"Failure rate (train): {y_train.mean():.2%}\n")

    results: dict[str, dict] = {}
    best_name, best_pipe, best_auc = None, None, -np.inf

    for name, model in candidate_models().items():
        pipe = Pipeline([("prep", preprocessor), ("clf", model)])
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        mean_auc = float(cv_auc.mean())
        results[name] = {"cv_roc_auc_mean": mean_auc, "cv_roc_auc_std": float(cv_auc.std())}
        print(f"{name:>20s}  CV ROC-AUC = {mean_auc:.4f} (+/- {cv_auc.std():.4f})")
        if mean_auc > best_auc:
            best_name, best_pipe, best_auc = name, pipe, mean_auc

    print(f"\nBest model: {best_name} (CV ROC-AUC = {best_auc:.4f})")

    # Refit the winner on the full training split and evaluate on the holdout.
    best_pipe.fit(X_train, y_train)
    proba = best_pipe.predict_proba(X_test)[:, 1]
    preds = best_pipe.predict(X_test)

    test_metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "average_precision": float(average_precision_score(y_test, proba)),
        "f1": float(f1_score(y_test, preds)),
    }
    print("\nHoldout test performance:")
    for k, v in test_metrics.items():
        print(f"  {k:>18s} = {v:.4f}")
    print("\n" + classification_report(y_test, preds, digits=3))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, MODEL_DIR / "failure_model.joblib")

    summary = {
        "best_model": best_name,
        "feature_columns": list(X.columns),
        "cross_validation": results,
        "holdout_test": test_metrics,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_failure_rate": float(y_train.mean()),
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved model -> {MODEL_DIR / 'failure_model.joblib'}")
    print(f"Saved metrics -> {MODEL_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
