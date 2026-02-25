import pandas as pd

def load_dataset(path):
    df = pd.read_excel(path)
    required_cols = ["TC", "TG", "HDL_C", "LDL_direct"]
    df = df.dropna(subset=required_cols)
    return df
