import pandas as pd
import numpy as np

def modify_Resolve_time(df: pd.DataFrame) -> pd.DataFrame:
    def adjust_time(row):
        if 0 <= row['Opened'].hour < 6:  
            return row['Resolve time'] * 0.8
        elif 22 <= row['Opened'].hour < 24:
            return row['Resolve time'] * 0.85
        else:
            return row['Resolve time']

    df['Opened'] = pd.to_datetime(df['Opened'])
    df['Resolve time'] = df.apply(adjust_time, axis=1)
    return df
if __name__ == "__main__":
   df = pd.read_csv("temp_data.csv")
   df = modify_Resolve time(df)
   df.to_csv("temp_data.csv", index=False)