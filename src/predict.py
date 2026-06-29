"""
Score new machine readings with the trained failure model.

Two modes:
  * --csv PATH   : score every row of a CSV with the feature columns and
                   write an output CSV with failure_probability + prediction.
  * (no args)    : run a built-in demo on a couple of hand-made examples.

    python src/predict.py --csv data/new_readings.csv --out scored.csv
    python src/predict.py            # demo

The probability threshold defaults to 0.5 but can be tuned (--threshold) to
trade recall for precision depending on maintenance cost tolerance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "failure_model.joblib"

FEATURE_COLUMNS = [
    "machine_quality",
    "air_temperature_K",
    "process_temperature_K",
    "rotational_speed_rpm",
    "torque_Nm",
    "tool_wear_min",
    "operating_hours",
    "vibration_mm_s",
    "pressure_bar",
]


def load_model():
    if not MODEL_PATH.exists():
        raise SystemExit("No trained model found - run `python src/train.py` first.")
    return joblib.load(MODEL_PATH)


def score(model, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"Input is missing required columns: {missing}")
    proba = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    out = df.copy()
    out["failure_probability"] = proba.round(4)
    out["predicted_failure"] = (proba >= threshold).astype(int)
    return out


def _demo_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # Healthy machine: cool, normal load, fresh tool.
            {
                "machine_quality": "H",
                "air_temperature_K": 299.0,
                "process_temperature_K": 309.5,
                "rotational_speed_rpm": 1550,
                "torque_Nm": 40.0,
                "tool_wear_min": 20,
                "operating_hours": 300.0,
                "vibration_mm_s": 1.3,
                "pressure_bar": 150.0,
            },
            # At-risk machine: worn tool, high torque -> overstrain regime.
            {
                "machine_quality": "L",
                "air_temperature_K": 301.0,
                "process_temperature_K": 309.0,
                "rotational_speed_rpm": 1320,
                "torque_Nm": 65.0,
                "tool_wear_min": 230,
                "operating_hours": 1500.0,
                "vibration_mm_s": 4.5,
                "pressure_bar": 138.0,
            },
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict mechanical failure.")
    parser.add_argument("--csv", type=Path, help="CSV of readings to score")
    parser.add_argument("--out", type=Path, help="where to write scored CSV")
    parser.add_argument("--threshold", type=float, default=0.5, help="decision threshold")
    args = parser.parse_args()

    model = load_model()

    if args.csv:
        df = pd.read_csv(args.csv)
        scored = score(model, df, args.threshold)
        if args.out:
            scored.to_csv(args.out, index=False)
            print(f"Wrote scored predictions -> {args.out}")
        else:
            print(scored.to_string(index=False))
    else:
        scored = score(model, _demo_frame(), args.threshold)
        cols = ["machine_quality", "tool_wear_min", "torque_Nm", "failure_probability", "predicted_failure"]
        print("Demo predictions:\n")
        print(scored[cols].to_string(index=False))


if __name__ == "__main__":
    main()
