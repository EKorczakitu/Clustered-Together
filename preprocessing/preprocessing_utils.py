import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Cleaning
# -----------------------------
def clean_data(df, drop_exp_above_1=True):
    df = df.copy()
    if drop_exp_above_1:
        df = df[df["Exposure"] <= 1]
    df = df.dropna()
    return df


# -----------------------------
# Feature selection (project choices)
# -----------------------------
def feature_selection(df):
    # Drop IDs and Density (we decided to keep Area and drop Density)
    drop_cols = [c for c in ["IDpol", "Density"] if c in df.columns]
    return df.drop(columns=drop_cols)


# -----------------------------
# Scaling numerical features (Prep for models that need scaling)
# -----------------------------
def scale_features(df, features):
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df, scaler

def seperate_features(df):
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
    
    return numeric_features, categorical_features

# -----------------------------
# Final preprocessing for M1 tree
# -----------------------------
def preprocess_for_tree(df):
    """
    Returns:
      X      -> one-hot features (no scaling)
      y_rate -> ClaimNb / Exposure
      w_expo -> Exposure
    """
    df = feature_selection(clean_data(df)).copy()

    y_rate = (df["ClaimNb"] / df["Exposure"]).astype(float)
    w_expo = df["Exposure"].astype(float)

    num_cols = [c for c in ["VehPower","VehAge","DrivAge","BonusMalus"] if c in df.columns]
    cat_cols = [c for c in ["Area","VehBrand","VehGas","Region"] if c in df.columns]

    X = pd.get_dummies(df[num_cols + cat_cols], columns=cat_cols, drop_first=True)
    return X, y_rate, w_expo


