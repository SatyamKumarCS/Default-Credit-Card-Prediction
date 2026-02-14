import pandas as pd

def load_data(path:str) -> pd.DataFrame:
    df=df.read_csv(path)
    df=df.rename(columns={'default payment next month':'Default'})
    df=df.drop("ID",axis=1)
    return df