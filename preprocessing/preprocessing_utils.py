# 1) Load data (adjust path if needed)
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")  # change if your CSVs live elsewhere
df = pd.read_csv(DATA_DIR / "claims_train.csv")

# 2) Minimal preprocessing for the tree (no scaling, OHE cats)
def clean_data(df, drop_exp_above_1=True):
    df = df.copy()
    if drop_exp_above_1:
        df = df[df["Exposure"] <= 1]
    df = df.dropna()
    return df

def feature_selection(df):
    # Drop IDs and Density (we decided to keep Area and drop Density)
    drop_cols = [c for c in ["IDpol", "Density"] if c in df.columns]
    return df.drop(columns=drop_cols)

def preprocess_for_tree(df):
    df = feature_selection(clean_data(df)).copy()
    y_rate = (df["ClaimNb"] / df["Exposure"]).astype(float)
    w_expo = df["Exposure"].astype(float)
    num_cols = [c for c in ["VehPower","VehAge","DrivAge","BonusMalus"] if c in df.columns]
    cat_cols = [c for c in ["Area","VehBrand","VehGas","Region"] if c in df.columns]
    X = pd.get_dummies(df[num_cols + cat_cols], columns=cat_cols, drop_first=True)
    return X, y_rate, w_expo

X, y_rate, w = preprocess_for_tree(df)

print("X shape:", X.shape)
print("Sample columns:", list(X.columns[:5]))
print("Targets/weights head:\n", pd.DataFrame({"y_rate": y_rate.head(), "Exposure": w.head()}))
