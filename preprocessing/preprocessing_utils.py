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
# Helper: Separate features
# -----------------------------
def seperate_features(df):
    """Separates numeric and categorical columns in a DataFrame."""
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
    return numeric_features, categorical_features

# -----------------------------
# Method 1: Preprocessing for Tree
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

# -----------------------------
# Method 2: Preprocessing for Neural Network (NEW)
# -----------------------------
def preprocess_for_nn(df, scaler=None, ref_columns=None):
    """
    Args:
        df: The dataframe to process
        scaler: (Optional) An existing StandardScaler object. 
                If None, a new one is fitted (use this for Training data).
                If provided, it transforms the data (use this for Test data).
        ref_columns: (Optional) List of column names from the training set 
                     to ensure the test set has the exact same columns/order.

    Returns:
        X      -> Scaled and One-Hot Encoded DataFrame
        y_rate -> ClaimNb / Exposure
        w_expo -> Exposure
        scaler -> The fitted scaler object (to be saved for the test set)
    """
    # 1. Clean and Select
    df = feature_selection(clean_data(df)).copy()

    # 2. Separation of Target and Weights
    y_rate = (df["ClaimNb"] / df["Exposure"]).astype(float)
    w_expo = df["Exposure"].astype(float)

    # 3. Define Feature Groups
    num_cols = ["VehPower", "VehAge", "DrivAge", "BonusMalus"]
    cat_cols = ["Area", "VehBrand", "VehGas", "Region"]

    # 4. Scaling Numerical Features
    # Crucial: Fit on Train, Transform on Test
    if scaler is None:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        df[num_cols] = scaler.transform(df[num_cols])

    # 5. One-Hot Encoding
    X = pd.get_dummies(df[num_cols + cat_cols], columns=cat_cols, drop_first=True)

    # 6. Ensure Columns Match (Handle missing/extra categories in Test)
    if ref_columns is not None:
        # Add missing columns with 0
        for col in ref_columns:
            if col not in X.columns:
                X[col] = 0
        # Drop extra columns not in train
        X = X[ref_columns]
        # Enforce order
        X = X[ref_columns]

    return X, y_rate, w_expo, scaler

def preprocess_actuarial(df, scaler=None, ref_columns=None):
    """
    Advanced preprocessing based on Actuarial literature.
    Features:
    - Density: Log-transformed
    - BonusMalus: Capped at 150, then Log-transformed
    - VehPower: Merged classes >= 9
    - VehAge, DrivAge: Binned into categories
    - Area: Mapped to integer (Ordinal) or kept categorical
    """
    # 1. Clean Data
    df = df.copy()
    df = df[df["Exposure"] <= 1].dropna()
    
    # 2. Targets
    y_rate = (df["ClaimNb"] / df["Exposure"]).astype(float)
    w_expo = df["Exposure"].astype(float)

    # -----------------------------
    # Feature Engineering
    # -----------------------------
    
    # A. Density (Log Transform)
    # We add 1 to avoid log(0) just in case, though density usually > 0
    df["LogDensity"] = np.log(df["Density"])
    
    # B. BonusMalus (Cap + Log)
    df["BonusMalus"] = df["BonusMalus"].clip(upper=150)
    df["LogBonusMalus"] = np.log(df["BonusMalus"])
    
    # C. VehPower (Merge >= 9)
    # We treat this as Categorical 
    df["VehPower_Binned"] = df["VehPower"].apply(lambda x: 9 if x >= 9 else x).astype(str)
    
    # D. VehAge (Binning)
    # Bins: [0, 1), [1, 10], (10, inf)
    # We use pd.cut. right=False means [0, 1), right=True means (0, 1]
    # Adjusting slightly to match your description:
    df["VehAge_Bin"] = pd.cut(df["VehAge"], 
                              bins=[-1, 0, 10, 100], 
                              labels=["New", "Medium", "Old"]).astype(str)

    # E. DrivAge (Binning)
    # Bins: [18, 21), [21, 26), [26, 31), [31, 41), [41, 51), [51, 71), [71, inf)
    age_bins = [17, 21, 26, 31, 41, 51, 71, 200]
    age_labels = ["18-21", "21-26", "26-31", "31-41", "41-51", "51-71", "71+"]
    df["DrivAge_Bin"] = pd.cut(df["DrivAge"], bins=age_bins, labels=age_labels).astype(str)
    
    # F. Area (Ordinal Encoding A=1, B=2...)
    area_map = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6}
    df["Area_Int"] = df["Area"].map(area_map)

    # -----------------------------
    # Scaling & Encoding
    # -----------------------------
    
    # Continuous Features to Scale
    # Note: We use the LOG versions, not the raw versions
    cont_cols = ["LogDensity", "LogBonusMalus", "Area_Int"] 
    
    if scaler is None:
        scaler = StandardScaler()
        df[cont_cols] = scaler.fit_transform(df[cont_cols])
    else:
        df[cont_cols] = scaler.transform(df[cont_cols])
        
    # Categorical Features to One-Hot
    # Note: We use the BINNED versions
    cat_cols = ["VehBrand", "VehGas", "Region", "VehPower_Binned", "VehAge_Bin", "DrivAge_Bin"]
    
    # Select final columns
    X = pd.get_dummies(df[cont_cols + cat_cols], columns=cat_cols, drop_first=True)
    
    # Ensure Columns Match (for Test set)
    if ref_columns is not None:
        for col in ref_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[ref_columns]
    
    return X, y_rate, w_expo, scaler