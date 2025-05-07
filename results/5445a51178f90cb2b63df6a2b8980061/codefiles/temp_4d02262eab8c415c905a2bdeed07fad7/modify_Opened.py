import pandas as pd
import numpy as np

def modify_Opened(df: pd.DataFrame) -> pd.DataFrame:
    conditions = (df['Category'] == 'Software') & (df['Opened'] >= pd.Timestamp('2023-10-01'))
    df.loc[conditions, 'Priority'] = 'New High-Impact'
    return df
if __name__ == "__main__":
   df = pd.read_csv("temp_data.csv")
   df = modify_Opened(df)
   df.to_csv("temp_data.csv", index=False)