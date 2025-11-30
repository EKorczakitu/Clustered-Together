import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. CORE CLEANING (Shared by all)
# ==========================================
def clean_data(df, drop_exp_above_1=True):
    """
    Basic row filtering.
    """
    df = df.copy()
    
    # 1. Cap Exposure at 1 (standard insurance practice)
    if drop_exp_above_1:
        df = df[[df["Exposure"] <= 1] & [df["Exposure"] > 0.01] ] # or replace with value of 1 instead of dropping
    
    # 2. Drop NaN
    df = df.dropna()
    
    # 3. Drop IDs (useless for prediction)
    if "IDpol" in df.columns:
        df = df.drop(columns=["IDpol"])
        
    return df

# ==========================================
# 2. METHOD M1: PREPROCESSING FOR TREES
# Strategy: Keep it RAW. Trees handle non-linearities and scaling themselves.
# ==========================================
def preprocess_for_tree(df):
    """
    Minimal preprocessing for Decision Trees / XGBoost / Random Forest.
    - No Scaling (Trees don't need it)
    - No Binning (Trees do this better automatically)
    - One-Hot Encoding for Categoricals
    """
    # 1. Clean
    df = clean_data(df)
    
    # 2. Separate Target & Weights
    y_rate = (df["ClaimNb"] / df["Exposure"]).astype(float)
    w_expo = df["Exposure"].astype(float)
    
    # 3. Define Features
    # Note: We keep Density and BonusMalus RAW. 
    # The Tree will find the split points (e.g. BonusMalus > 105).
    num_cols = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
    cat_cols = ["Area", "VehBrand", "VehGas", "Region"]
    
    # 4. One-Hot Encoding / no because no one hot encoding for trees according to book
    # # We drop_first=True to reduce multicollinearity, though trees handle it okay.
    # X = pd.get_dummies(df[num_cols + cat_cols], columns=cat_cols, drop_first=True)
    X= df[num_cols + cat_cols]
    return X, y_rate, w_expo


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def nn_preprocess(df, scaler=None, ref_columns=None):
    # -----------------------------
    # 1. Basic cleaning
    # -----------------------------
    df = clean_data(df)  # Your own cleaning function (Exposure cap, ClaimNb cap)

    # Targets
    y_rate = (df["ClaimNb"] / df["Exposure"]).astype(float)
    w_expo = df["Exposure"].astype(float)

    # -----------------------------
    # 2. Log-transform skewed continuous predictors
    # -----------------------------
    df["LogDensity"] = np.log1p(df["Density"])           # continuous, heavily skewed
    df["LogBonusMalus"] = np.log(df["BonusMalus"])       # continuous, multiplicative risk

    # -----------------------------
    # 3. Continuous columns for scaling
    # -----------------------------
    cont_cols = ["LogDensity", "LogBonusMalus", "DrivAge"]

    # Standardize continuous features (ISL requirement)
    if scaler is None:
        scaler = StandardScaler()
        df[cont_cols] = scaler.fit_transform(df[cont_cols])
    else:
        df[cont_cols] = scaler.transform(df[cont_cols])

    # -----------------------------
    # 4. Categorical variables (dummy-encoding)
    # -----------------------------
    cat_cols = ["Area", "VehBrand", "VehGas", "Region", "VehPower", "VehAge"]

    # Create final model matrix
    X = pd.get_dummies(df[cont_cols + cat_cols], columns=cat_cols, drop_first=True)

    # -----------------------------
    # 5. Column alignment (Train/Test consistency)
    # -----------------------------
    if ref_columns is not None:

        # Add missing columns from training set
        for col in ref_columns:
            if col not in X.columns:
                X[col] = 0

        # Remove any extra columns not seen during training
        X = X[ref_columns]

    return X, y_rate, w_expo, scaler