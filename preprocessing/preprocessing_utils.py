import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean_data(df, drop_exp_above_1=True):
    if drop_exp_above_1:
        df = df[df["Exposure"] <= 1]
    df = df.dropna()
    return df

def encode_categoricals(df, one_hot=True):
    cat_cols = ["VehBrand", "VehGas", "Area", "Region"]
    if one_hot:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    else:
        for c in cat_cols:
            df[c] = df[c].astype("category").cat.codes
    return df

def scale_features(df, features):
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df, scaler
