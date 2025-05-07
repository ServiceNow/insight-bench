import pandas as pd
import numpy as np

def modify_Priority(df: pd.DataFrame) -> pd.DataFrame:
    df['Priority'] = df['Priority'].apply(lambda x: 1 if x.lower() in ['high', 'urgent'] else 3)
    df['Priority'] = df['Priority'].astype(int)
    cascade_effect = df['Priority'].rolling(window=3, min_periods=1).mean()
    df.loc[cascade_effect > 2, 'Priority'] = 2
    return df
if __name__ == "__main__":
   df = pd.read_csv("temp_data.csv")
   df = modify_Priority(df)
   df.to_csv("temp_data.csv", index=False)