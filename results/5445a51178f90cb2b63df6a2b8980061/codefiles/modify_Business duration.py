import pandas as pd
import numpy as np

def modify_Business_duration(df: pd.DataFrame) -> pd.DataFrame:
    df['month'] = pd.to_datetime(df['date']).dt.month
    df['Business duration'] = df.apply(lambda row: row['Business duration'] * 1.5 if row['month'] == 12 else (row['Business duration'] * 0.7 if row['month'] == 2 else row['Business duration']), axis=1)
    df['Resolve time'] = df.apply(lambda row: row['Resolve time'] * 1.5 if row['month'] == 12 else (row['Resolve time'] * 0.7 if row['month'] == 2 else row['Resolve time']), axis=1)
    return df.drop(columns='month')
if __name__ == "__main__":
   df = pd.read_csv("temp_data.csv")
   df = modify_Business duration(df)
   df.to_csv("temp_data.csv", index=False)