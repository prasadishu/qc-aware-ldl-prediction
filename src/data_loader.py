import pandas as pd

def load_dataset(path):
    df = pd.read_excel(path)
    df = df.dropna()
    return df
