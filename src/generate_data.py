"""
Synthetic accident / mechanical-failure data generator.

No real accident dataset was supplied with this project, so this module
synthesises a physically-plausible one. Each row is a snapshot of a machine's
operating condition together with whether it failed and, if so, the failure
mode. The failure logic mirrors the well-documented behaviour of industrial
predictive-maintenance data (e.g. the AI4I 2020 dataset): failures are driven
by interpretable physical relationships plus a small amount of random noise,
so a model has real signal to learn rather than pure randomness.

Run directly to (re)generate ``data/accidents.csv``:

    python src/generate_data.py --rows 12000 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Quality variants of the product/machine and their relative frequency.
# Higher quality -> sturdier -> higher overstrain tolerance.
QUALITY_LEVELS = ["L", "M", "H"]
QUALITY_WEIGHTS = [0.50, 0.30, 0.20]
# Tool-wear * torque overstrain threshold (N.m.min) per quality level.
OSF_THRESHOLD = {"L": 11_000, "M": 12_000, "H": 13_000}


def _simulate(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Draw n machine snapshots with correlated, realistic sensor readings."""
    quality = rng.choice(QUALITY_LEVELS, size=n, p=QUALITY_WEIGHTS)

    # Ambient air temperature, K (~300 K with mild spread).
    air_temp = rng.normal(300.0, 2.0, n)
    # Process temperature sits ~10 K above air temp plus its own noise.
    process_temp = air_temp + rng.normal(10.0, 1.0, n)

    # Torque, N.m, and rotational speed, rpm (mild inverse load relation).
    torque = np.clip(rng.normal(40.0, 10.0, n), 3.5, 76.0)
    rot_speed = 1500.0 + 8.0 * (40.0 - torque) + rng.normal(0.0, 90.0, n)
    rot_speed = np.clip(rot_speed, 1168.0, None)

    # Cumulative tool wear, minutes.
    tool_wear = rng.integers(0, 250, n).astype(float)

    # Operating hours since last maintenance.
    operating_hours = rng.gamma(shape=2.0, scale=400.0, size=n)

    # Vibration, mm/s. Rises with tool wear and torque (mechanical stress).
    vibration = (
        0.8
        + 0.010 * tool_wear
        + 0.020 * torque
        + rng.normal(0.0, 0.3, n)
    )
    vibration = np.clip(vibration, 0.1, None)

    # Hydraulic/working pressure, bar.
    pressure = rng.normal(150.0, 12.0, n)

    return pd.DataFrame(
        {
            "machine_quality": quality,
            "air_temperature_K": np.round(air_temp, 2),
            "process_temperature_K": np.round(process_temp, 2),
            "rotational_speed_rpm": np.round(rot_speed, 1),
            "torque_Nm": np.round(torque, 2),
            "tool_wear_min": tool_wear.astype(int),
            "operating_hours": np.round(operating_hours, 1),
            "vibration_mm_s": np.round(vibration, 3),
            "pressure_bar": np.round(pressure, 2),
        }
    )


def _label_failures(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Assign a failure flag and failure mode using physical heuristics."""
    n = len(df)
    # Mechanical power, W: torque (N.m) * angular velocity (rad/s).
    power = df["torque_Nm"].to_numpy() * (df["rotational_speed_rpm"].to_numpy() * 2 * np.pi / 60.0)
    temp_gap = (df["process_temperature_K"] - df["air_temperature_K"]).to_numpy()
    osf_strain = df["tool_wear_min"].to_numpy() * df["torque_Nm"].to_numpy()
    osf_limit = df["machine_quality"].map(OSF_THRESHOLD).to_numpy()

    # Tool Wear Failure: heavily worn tools fail in a small random fraction.
    tool_wear = df["tool_wear_min"].to_numpy()
    twf = (tool_wear >= 200) & (tool_wear <= 240) & (rng.random(n) < 0.10)
    # Heat Dissipation Failure: poor cooling at low speed.
    hdf = (temp_gap < 8.6) & (df["rotational_speed_rpm"].to_numpy() < 1380)
    # Power Failure: power outside the safe operating envelope.
    pwf = (power < 3500) | (power > 9000)
    # Overstrain Failure: wear*torque exceeds quality-dependent limit.
    osf = osf_strain > osf_limit
    # Random Failure: rare, unexplained.
    rnf = rng.random(n) < 0.002

    failure_type = np.full(n, "No Failure", dtype=object)
    # Precedence order: assign the first matching, most-specific mode.
    for mask, name in [
        (hdf, "Heat Dissipation Failure"),
        (pwf, "Power Failure"),
        (osf, "Overstrain Failure"),
        (twf, "Tool Wear Failure"),
        (rnf, "Random Failure"),
    ]:
        failure_type = np.where((failure_type == "No Failure") & mask, name, failure_type)

    df = df.copy()
    df["failure_type"] = failure_type
    df["machine_failure"] = (failure_type != "No Failure").astype(int)
    return df


def generate(n: int = 12_000, seed: int = 42) -> pd.DataFrame:
    """Generate a labelled mechanical-failure dataset of ``n`` rows."""
    rng = np.random.default_rng(seed)
    df = _simulate(n, rng)
    df = _label_failures(df, rng)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic failure data.")
    parser.add_argument("--rows", type=int, default=12_000, help="number of rows")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "accidents.csv",
        help="output CSV path",
    )
    args = parser.parse_args()

    df = generate(args.rows, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    rate = df["machine_failure"].mean()
    print(f"Wrote {len(df):,} rows to {args.out}")
    print(f"Overall failure rate: {rate:.2%}")
    print("Failure mode breakdown:")
    print(df["failure_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
