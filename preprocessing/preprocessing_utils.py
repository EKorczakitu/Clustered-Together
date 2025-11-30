import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

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
    df = df.copy()
    
    # 1. Cap Exposure
    df["Exposure"] = df["Exposure"].clip(upper=1.0)
    
    # 2. Drop NaN and IDs
    df = df.dropna()
    if "IDpol" in df.columns:
        df = df.drop(columns=["IDpol"])
    
    # 3. Define Targets
    y_rate = (df["ClaimNb"] / df["Exposure"]).astype(float)
    w_expo = df["Exposure"].astype(float)
    
    # 4. Feature Engineering
    
    # A. AREA: Ordinal Encoding (Better than One-Hot for Trees)
    # This fixes the "ValueError: string to float" for Area
    area_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6}
    df["Area_Int"] = df["Area"].map(area_map)
    
    # B. OTHER CATEGORICALS: One-Hot Encoding
    # Sklearn cannot handle strings, so we must encode Region/Brand.
    # We drop_first=True to keep it cleaner, though trees handle multicollinearity fine.
    cats_to_encode = ["VehBrand", "VehGas", "Region"]
    
    # Keep numeric columns + the new Ordinal Area
    keep_cols = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density", "Area_Int"]
    
    # Create final X
    # This converts "Region" -> "Region_R24", "Region_R25"... which are numbers (0/1)
    X = pd.get_dummies(df[keep_cols + cats_to_encode], columns=cats_to_encode, drop_first=True)
    
    return X, y_rate, w_expo


def nn_preprocess(df, scaler=None, ref_columns=None):
    # -----------------------------
    # 1. Basic cleaning
    # -----------------------------
    # 1. Clean
    df = df.copy()
    
    # 1. Cap Exposure at 1 (standard insurance practice)
    df["Exposure"] = df["Exposure"].clip(upper=1.0)
    
    # 2. Drop NaN
    df = df.dropna()
    
    # 3. Drop IDs (useless for prediction)
    if "IDpol" in df.columns:
        df = df.drop(columns=["IDpol"])

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
    
    # ======================================================
    # 5. ALIGNMENT FIX (Crucial Step)
    # ======================================================
    if ref_columns is not None:
        # Add missing columns with 0
        for col in ref_columns:
            if col not in X.columns:
                X[col] = 0
        
        # Drop extra columns (if categories appear in Test but not Train)
        X = X[ref_columns]
        
        # Ensure exact order matches Train
        X = X[ref_columns]

    return X, y_rate, w_expo, scaler

    return X, y_rate, w_expo, scaler