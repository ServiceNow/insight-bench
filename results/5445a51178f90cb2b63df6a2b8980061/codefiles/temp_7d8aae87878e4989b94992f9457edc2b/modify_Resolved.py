import pandas as pd
import numpy as np

def modify_Resolved(df: pd.DataFrame) -> pd.DataFrame:
    if 'Opened' not in df.columns or 'Resolved' not in df.columns:
        return df
    df['Opened'] = pd.to_datetime(df['Opened'], errors='coerce')
    df['Resolved'] = pd.to_datetime(df['Resolved'], errors='coerce')
    off_peak_hours = df['Opened'].dt.hour.isin(range(0, 6)) | df['Opened'].dt.hour.isin(range(22, 24))
    df.loc[off_peak_hours & df['Resolved'].isna(), 'Resolved'] = df.loc[off_peak_hours, 'Opened'] + pd.Timedelta(hours=1)
    return df
if __name__ == "__main__":
   df = pd.read_csv("temp_data.csv")
   df = modify_Resolved(df)
   df.to_csv("temp_data.csv", index=False)