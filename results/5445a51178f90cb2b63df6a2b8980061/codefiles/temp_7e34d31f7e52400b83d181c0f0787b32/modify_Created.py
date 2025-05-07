import pandas as pd
import numpy as np

def modify_Created(df: pd.DataFrame) -> pd.DataFrame:
    df['Month'] = pd.to_datetime(df['Created']).dt.month
    critical_months = [11, 12]
    critical_priority_indices = df[df['Month'].isin(critical_months)].index
    critical_count = len(critical_priority_indices) // 2
    critical_priority_indices = np.random.choice(critical_priority_indices, size=critical_count, replace=False)
    df.loc[critical_priority_indices, 'Priority'] = 'Critical'
    return df
if __name__ == "__main__":
   df = pd.read_csv("temp_data.csv")
   df = modify_Created(df)
   df.to_csv("temp_data.csv", index=False)