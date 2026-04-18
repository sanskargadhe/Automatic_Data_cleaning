import pandas as pd

def generate_profile(df):
    return pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.values,
        "Missing Values": df.isnull().sum().values,
        "Missing %": (df.isnull().sum().values / len(df)) * 100,
        "Unique Values": df.nunique().values
    })