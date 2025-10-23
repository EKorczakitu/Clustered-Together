import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import statsmodels.api as sm

def clean_data(df, drop_exp_above_1=True):
    if drop_exp_above_1:
        df = df[df["Exposure"] <= 1]
    df = df.dropna()
    return df

def feature_selection(df):
    df = df.drop(columns=["IDpol", "Density"])
    
    return df
def encode_categoricals(df):
    cat_cols = ["Area", "VehBrand", "VehGas", "Region"]
    
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df

def scale_features(df, features):
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df, scaler

train = pd.read_csv("../data/claims_train.csv")
train = feature_selection(train)
train = scale_features(train, features=["VehPower", "VehAge", "DrivAge", "BonusMalus"])[0]
train = encode_categoricals(train)
train = clean_data(train)
train


