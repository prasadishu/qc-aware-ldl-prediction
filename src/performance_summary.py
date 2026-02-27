import pandas as pd

def save_performance(results):
    df = pd.DataFrame(results)
    df.to_excel("outputs/tables/Performance_Summary.xlsx", index=False)
    
