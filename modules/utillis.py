def data_quality_score(df):
    total = df.size
    missing = df.isnull().sum().sum()
    return round((1 - missing/total)*100, 2)