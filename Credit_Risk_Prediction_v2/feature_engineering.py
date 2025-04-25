def engineer_features(df):
    # for new feature
    df["loan_to_income"] = df["Credit amount"] / df["Duration"]
    return df
