import pandas as pd
import numpy as np
from scipy.stats import zscore

def clean_data(df):
    df = df.copy()

    # Handle missing values
    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)


    # Remove outliers
    numeric_cols = df.select_dtypes(include=['int64','float64']).columns
    if len(numeric_cols) > 0:
        df = df[(np.abs(zscore(df[numeric_cols])) < 3).all(axis=1)]

    return df