"""
clean.py — Clean train.csv based on the 7 discovered data problems.

Fixes applied:
  1. Drop leaking column (metres_checked_at_folding)
  2. Fix unit_mixup: power_cuts_min for Shed 1 is in seconds → divide by 60
  3. Fix sentinel_value: loom_age_yr == 0 → set to NaN, then impute with median
  4. Drop duplicate rows (keep first)
  5. Fix stuck_sensor: shift_hours stuck at 10.8 from 2026-03-01 to 2026-03-20
     → set to NaN, forward-fill from surrounding dates (or median impute)
  6. target_typos: not present — no action
  7. spelling_variants: not present — no action

Run:  python src/mp026/clean.py
"""
import os
import pandas as pd
import numpy as np
import yaml

PARAMS_PATH = "params.yaml"
TRAIN_PATH  = "data/raw/train.csv"
OUT_DIR     = "data/processed"
OUT_PATH    = "data/processed/train_clean.csv"


def load_params():
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Drop leaking column
    if "metres_checked_at_folding" in df.columns:
        df = df.drop(columns=["metres_checked_at_folding"])

    # 2. Fix unit mixup: Shed 1 recorded power_cuts_min in seconds
    shed1_mask = df["shed"] == "Shed 1"
    df.loc[shed1_mask, "power_cuts_min"] = df.loc[shed1_mask, "power_cuts_min"] / 60

    # 3. Fix sentinel: loom_age_yr == 0 → NaN → median
    df.loc[df["loom_age_yr"] == 0, "loom_age_yr"] = np.nan
    df["loom_age_yr"] = df["loom_age_yr"].fillna(df["loom_age_yr"].median())

    # 4. Drop duplicate rows (keep first occurrence)
    df = df.drop_duplicates().reset_index(drop=True)

    # 5. Fix stuck sensor: shift_hours stuck at 10.8 from 2026-03-01 to 2026-03-20
    stuck_mask = (
        (df["date"] >= "2026-03-01") &
        (df["date"] <= "2026-03-20") &
        (df["shift_hours"] == 10.8)
    )
    df.loc[stuck_mask, "shift_hours"] = np.nan
    # sort by date and forward-fill, then back-fill any remaining NaN
    df = df.sort_values("date").reset_index(drop=True)
    df["shift_hours"] = df["shift_hours"].ffill().bfill()

    # 6 & 7: target_typos and spelling_variants not present — no action

    return df


def main():
    params = load_params()
    os.makedirs(OUT_DIR, exist_ok=True)

    train = pd.read_csv(TRAIN_PATH)
    print(f"Original shape: {train.shape}")

    clean_df = clean(train)
    print(f"Cleaned shape: {clean_df.shape}")

    clean_df.to_csv(OUT_PATH, index=False)
    print(f"Saved to {OUT_PATH}")

    # Quick sanity
    print("\nSanity checks after cleaning:")
    print(f"  Duplicate rows: {clean_df.duplicated().sum()}")
    print(f"  loom_age_yr == 0: {(clean_df['loom_age_yr'] == 0).sum()}")
    print(f"  Shed 1 power_cuts_min max: {clean_df[clean_df['shed']=='Shed 1']['power_cuts_min'].max():.1f} (should be <= 120)")
    print(f"  shift_hours NaN: {clean_df['shift_hours'].isna().sum()}")


if __name__ == "__main__":
    main()
