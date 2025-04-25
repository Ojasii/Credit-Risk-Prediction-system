import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(df):
    
    # drop unnecessary columns
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # missing values
    df['Saving accounts'].fillna('none', inplace=True)
    df['Checking account'].fillna('none', inplace=True)

    # Encoding categorical variables
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    # Scaling numerical features
    df['Risk'] = (df['Credit amount'] > 5000).astype(int)

    # 1 = bad risk (Credit > 5000), 0 = good risk
    return df
