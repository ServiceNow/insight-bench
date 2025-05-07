import pandas as pd
import numpy as np
from datetime import timedelta

def modify_Resolved(df: pd.DataFrame) -> pd.DataFrame:
    if 'Resolved' in df:
        resolved_dates = pd.to_datetime(df['Resolved'], errors='coerce')
        df['Resolved'] = resolved_dates
        spike_date = resolved_dates.mode()[0]
        spike_period = (resolved_dates >= spike_date) & (resolved_dates < spike_date + timedelta(days=3))
        df.loc[spike_period, 'Resolution Time'] = df.loc[spike_period, 'Resolution Time'] * 1.5
    return df
if __name__ == "__main__":
   df = pd.read_csv("temp_data.csv")
   df = modify_Resolved(df)
   df.to_csv("temp_data.csv", index=False)