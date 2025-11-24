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
        df = df[df["Exposure"] <= 1] # or replace with value of 1 instead of dropping
    
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
    
    # 4. One-Hot Encoding
    # We drop_first=True to reduce multicollinearity, though trees handle it okay.
    X = pd.get_dummies(df[num_cols + cat_cols], columns=cat_cols, drop_first=True)
    
    return X, y_rate, w_expo

# ==========================================
# 3. METHOD M2/M3: ACTUARIAL PREPROCESSING
# Strategy: Feature Engineering for Linear/Neural Models.
# - Log-transforms for heavy tails (Density)
# - Binning for U-shaped risks (Age)
# - Scaling for convergence
# ==========================================
def preprocess_actuarial(df, scaler=None, ref_columns=None):
    """
    Advanced preprocessing for Neural Networks (M2) or GLMs (M3).
    
    Args:
        scaler: StandardScaler object (Fit on Train, Transform on Test)
        ref_columns: List of columns from Train to ensure Test has exact match
    """
    # 1. Clean
    df = clean_data(df)
    
    # 2. Target & Weights
    y_rate = (df["ClaimNb"] / df["Exposure"]).astype(float)
    w_expo = df["Exposure"].astype(float)

    # --- Feature Engineering ---

    # A. Density -> Log Transform
    # Density spans 0 to 20,000+. Log compresses this range.
    # adding 1 is safe practice for log(0), though density usually > 0
    df["LogDensity"] = np.log1p(df["Density"]) 
    
    # B. BonusMalus -> Cap & Log
    # Capping at 150 prevents extreme "bad drivers" from skewing the mean
    df["BonusMalus"] = df["BonusMalus"].clip(upper=150)
    df["LogBonusMalus"] = np.log(df["BonusMalus"])
    
    # C. VehPower -> Group High Power
    # Cars with power > 9 are rare and behave similarly (sports cars/luxury)
    df["VehPower_Binned"] = df["VehPower"].apply(lambda x: 9 if x >= 9 else x).astype(str)
    
    # D. VehAge -> Binned
    # Old cars are safer (less driven/careful owners) vs New cars.
    # Using 'cut' ensures we handle the continuous nature correctly.
    df["VehAge_Bin"] = pd.cut(df["VehAge"], 
                              bins=[-1, 0, 10, 100], 
                              labels=["New", "Medium", "Old"]).astype(str)

    # E. DrivAge -> Binned (Actuarial Standard)
    # This captures the "Young Driver Risk" without the model needing to learn a complex non-linear curve.
    age_bins = [17, 21, 26, 31, 41, 51, 71, 200]
    age_labels = ["18-21", "21-26", "26-31", "31-41", "41-51", "51-71", "71+"]
    df["DrivAge_Bin"] = pd.cut(df["DrivAge"], bins=age_bins, labels=age_labels).astype(str)
    
    # F. Area -> Ordinal (Integer)
    # Area is ordinal (A is rural, F is Paris). The order matters.
    area_map = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6}
    df["Area_Int"] = df["Area"].map(area_map)

    # --- Scaling & Encoding ---
    
    # Continuous features (Using the Transformed versions)
    cont_cols = ["LogDensity", "LogBonusMalus", "Area_Int"]
    
    # Initialize or Apply Scaler
    if scaler is None:
        scaler = StandardScaler()
        df[cont_cols] = scaler.fit_transform(df[cont_cols])
    else:
        df[cont_cols] = scaler.transform(df[cont_cols])
        
    # Categorical features
    cat_cols = ["VehBrand", "VehGas", "Region", "VehPower_Binned", "VehAge_Bin", "DrivAge_Bin"]
    
    # Create final X
    X = pd.get_dummies(df[cont_cols + cat_cols], columns=cat_cols, drop_first=True)
    
    # --- Alignment (Crucial for Test Set) ---
    if ref_columns is not None:
        # 1. Add missing columns (filled with 0)
        for col in ref_columns:
            if col not in X.columns:
                X[col] = 0
        # 2. Remove extra columns (e.g., a region present in Test but not Train)
        X = X[ref_columns]
        # 3. Enforce order
        X = X[ref_columns]
    
    return X, y_rate, w_expo, scaler