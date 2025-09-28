from sklearn.model_selection import train_test_split
import pandas as pd

def prepare_data(df, target_col):
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # from string to numbers
    X = pd.get_dummies(X, drop_first=True)

    return train_test_split(X, y, test_size=0.2, random_state=42)