"""
Generate line graphs (risk curves) for the trained failure model and save
them to ``reports/figures/`` as PNG files.

Each curve is a *partial-dependence-style* line: one sensor is swept across
its observed range while every other feature is held at its median, and the
model's predicted failure probability is plotted as a line. These answer the
practical question "as this reading rises, how does failure risk move?".

Also produces an empirical line graph of failure rate vs operating-hours bins
straight from the historical data (no model needed).

    python src/graphs.py

Files written:
  reports/figures/risk_curve_tool_wear_min.png
  reports/figures/risk_curve_torque_Nm.png
  reports/figures/risk_curve_rotational_speed_rpm.png
  reports/figures/risk_curve_process_temperature_K.png
  reports/figures/failure_rate_vs_operating_hours.png
  reports/figures/risk_curves_combined.png
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "accidents.csv"
MODEL_PATH = ROOT / "models" / "failure_model.joblib"
FIG_DIR = ROOT / "reports" / "figures"

# Sensors to sweep, with friendly axis labels.
SWEEP_FEATURES = {
    "tool_wear_min": "Tool wear (min)",
    "torque_Nm": "Torque (N·m)",
    "rotational_speed_rpm": "Rotational speed (rpm)",
    "process_temperature_K": "Process temperature (K)",
}
N_POINTS = 120


def _baseline_row(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """A single representative machine: median numeric, modal categorical."""
    row = {}
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            row[col] = df[col].median()
        else:
            row[col] = df[col].mode().iloc[0]
    return pd.DataFrame([row])


def _risk_curve(model, df, feature_cols, feature):
    """Return (x grid, predicted failure probability) sweeping one feature."""
    lo, hi = df[feature].min(), df[feature].max()
    grid = np.linspace(lo, hi, N_POINTS)
    base = _baseline_row(df, feature_cols)
    sweep = pd.concat([base] * N_POINTS, ignore_index=True)
    sweep[feature] = grid
    proba = model.predict_proba(sweep[feature_cols])[:, 1]
    return grid, proba


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit("No trained model found - run `python src/train.py` first.")

    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    feature_cols = [c for c in df.columns if c not in ("machine_failure", "failure_type")]
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []

    # One individual line graph per swept sensor.
    for feature, label in SWEEP_FEATURES.items():
        grid, proba = _risk_curve(model, df, feature_cols, feature)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(grid, proba, color="#c0392b", linewidth=2.2)
        ax.fill_between(grid, proba, alpha=0.12, color="#c0392b")
        ax.set_xlabel(label)
        ax.set_ylabel("Predicted failure probability")
        ax.set_ylim(0, 1)
        ax.set_title(f"Failure risk vs {label.lower()}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = FIG_DIR / f"risk_curve_{feature}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        saved.append(out)

    # Combined multi-line graph (normalised x so curves share one axis).
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for feature, label in SWEEP_FEATURES.items():
        grid, proba = _risk_curve(model, df, feature_cols, feature)
        x_norm = (grid - grid.min()) / (grid.max() - grid.min())
        ax.plot(x_norm, proba, linewidth=2, label=label)
    ax.set_xlabel("Sensor value (normalised 0–1 over observed range)")
    ax.set_ylabel("Predicted failure probability")
    ax.set_ylim(0, 1)
    ax.set_title("Failure risk curves by sensor")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "risk_curves_combined.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    saved.append(out)

    # Empirical line graph: actual failure rate vs operating-hours bins.
    bins = pd.qcut(df["operating_hours"], q=12, duplicates="drop")
    grouped = df.groupby(bins, observed=True)["machine_failure"].mean()
    centers = [iv.mid for iv in grouped.index]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(centers, grouped.values * 100, marker="o", color="#2c3e50", linewidth=2)
    ax.set_xlabel("Operating hours since maintenance (bin center)")
    ax.set_ylabel("Observed failure rate (%)")
    ax.set_title("Historical failure rate vs operating hours")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "failure_rate_vs_operating_hours.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    saved.append(out)

    print(f"Saved {len(saved)} line graphs to {FIG_DIR}:")
    for p in saved:
        print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
