def get_features_and_target(df):
    """Splits the dataframe into inputs (X) and target (y)."""
    X = df.drop(columns=['target'])
    y = df['target']
    return X, y
