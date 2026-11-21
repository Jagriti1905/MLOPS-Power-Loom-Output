
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_FILE = BASE_DIR / "data" / "raw" / "train.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "train_cleaned.csv"


def clean_data():
    # Load original training data
    df = pd.read_csv(RAW_FILE)

    print("Original shape:", df.shape)

    # 1. Remove data leakage column
    if "metres_checked_at_folding" in df.columns:
        df = df.drop(columns=["metres_checked_at_folding"])

    # 2. Remove exact duplicate rows
    df = df.drop_duplicates()

    # 3. Correct power cuts unit for Shed 1
    mask = df["shed"] == "Shed 1"
    df.loc[mask, "power_cuts_min"] = (
        df.loc[mask, "power_cuts_min"] / 60
    )

    # 4. Document stuck sensor issue
    stuck_mask = (
        (df["date"] >= "2026-03-01") &
        (df["date"] <= "2026-03-20") &
        (df["shift_hours"] == 10.8)
    )

    print("Stuck sensor rows retained:", stuck_mask.sum())

    # Save cleaned data
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("Cleaned shape:", df.shape)
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    clean_data()
