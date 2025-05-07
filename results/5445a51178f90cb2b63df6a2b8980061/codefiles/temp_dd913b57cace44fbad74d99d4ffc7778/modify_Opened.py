import pandas as pd
import numpy as np

def modify_Opened(df: pd.DataFrame) -> pd.DataFrame:
    df['Opened'] = pd.to_datetime(df['Opened'])
    df['IsFirstWeekOfQuarter'] = ((df['Opened'].dt.month % 3 == 1) & (df['Opened'].dt.day <= 7))
    df['Incident Severity'] = np.where(df['IsFirstWeekOfQuarter'], '1 - High', 'Normal')
    df.drop(columns='IsFirstWeekOfQuarter', inplace=True)
    return df
if __name__ == "__main__":
   df = pd.read_csv("temp_data.csv")
   df = modify_Opened(df)
   df.to_csv("temp_data.csv", index=False)