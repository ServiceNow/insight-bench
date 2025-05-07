import pandas as pd
import numpy as np

def modify_Priority(df: pd.DataFrame) -> pd.DataFrame:
    df['Opened'] = pd.to_datetime(df['Opened'])
    df['Resolved'] = pd.to_datetime(df['Resolved'])
    df['is_off_hours'] = df['Opened'].dt.hour.isin(range(0, 9)) | df['Opened'].dt.hour.isin(range(18, 24))
    df['time_to_resolution'] = (df['Resolved'] - df['Opened']).dt.total_seconds() / 3600.0
    off_hours_delay = df.loc[df['is_off_hours'], 'time_to_resolution'].mean()
    df['Priority'] = np.where(df['is_off_hours'], df['time_to_resolution'] - off_hours_delay, df['time_to_resolution'])
    df['Priority'] = np.where(df['Priority'] < 0, 0, df['Priority'])
    return df.drop(columns=['is_off_hours', 'time_to_resolution'])
if __name__ == "__main__":
   df = pd.read_csv("temp_data.csv")
   df = modify_Priority(df)
   df.to_csv("temp_data.csv", index=False)