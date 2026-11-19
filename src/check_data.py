
import pandas as pd
import numpy as np

# Load datasets
train = pd.read_csv("data/raw/train.csv")
later = pd.read_csv("data/raw/later.csv")
holdout = pd.read_csv("data/raw/holdout_features.csv")

print("TRAIN SHAPE:", train.shape)
print("\nCOLUMNS:")
print(train.columns.tolist())

# 1. LEAKING COLUMN
print("\n--- 1. LEAKING COLUMN ---")
print("Train columns missing in holdout:")
print(set(train.columns) - set(holdout.columns))

if "metres_per_shift" in train.columns:
    print("\nCorrelation with target:")
    print(train.select_dtypes(include=np.number).corr()["metres_per_shift"].sort_values(ascending=False))

# 2. UNIT MIXUP
print("\n--- 2. UNIT MIXUP ---")
numeric_cols = train.select_dtypes(include=np.number).columns

for col in numeric_cols:
    if col == "metres_per_shift":
        continue
    print(f"\nColumn: {col}")
    print(train.groupby("shed")[col].agg(["min", "mean", "max"]))

# 3. SENTINEL VALUE
print("\n--- 3. SENTINEL VALUE ---")
for col in numeric_cols:
    counts = train[col].value_counts()
    unusual = counts[counts.index.isin([-9999, -999, -1, 0, 999, 9999])]
    if not unusual.empty:
        print(col, unusual.to_dict())

# 4. DUPLICATE ROWS
print("\n--- 4. DUPLICATE ROWS ---")
print("Duplicate rows:", train.duplicated().sum())

# 5. STUCK SENSOR
print("\n--- 5. STUCK SENSOR ---")
sorted_train = train.sort_values("date").reset_index(drop=True)

for col in numeric_cols:
    values = sorted_train[col]
    runs = values.groupby(values.ne(values.shift()).cumsum()).agg(["size"])
    longest = runs["size"].max()

    if longest >= 5:
        group_id = runs["size"].idxmax()
        indexes = np.where(
            values.ne(values.shift()).cumsum() == group_id
        )[0]

        if len(indexes) >= 5:
            start = indexes[0]
            end = indexes[-1]
            print(
                col,
                "Value:", values.iloc[start],
                "Count:", len(indexes),
                "Start:", sorted_train["date"].iloc[start],
                "End:", sorted_train["date"].iloc[end]
            )

# 6. TARGET TYPOS
print("\n--- 6. TARGET TYPOS ---")
target = train["metres_per_shift"].sort_values()
print(target.to_string(index=False))

# 7. SPELLING VARIANTS
print("\n--- 7. SPELLING VARIANTS ---")
text_cols = train.select_dtypes(include="object").columns

for col in text_cols:
    print(f"\n{col} - TRAIN:")
    print(train[col].value_counts(dropna=False))

    if col in later.columns:
        print(f"{col} - LATER:")
        print(later[col].value_counts(dropna=False))

print("\nDATA ANALYSIS COMPLETED!")
