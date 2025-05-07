import pandas as pd
import numpy as np
from datetime import timedelta

def modify_Priority(df: pd.DataFrame) -> pd.DataFrame:
    cutoff_date = pd.Timestamp('2023-01-01')
    df['Escalate'] = np.where(
        (df['Category'] == 'Network') & 
        (df['Created_Date'] < cutoff_date) &
        (pd.Timestamp('now') - df['Created_Date'] <= timedelta(days=1)), 
        True, False)
    df.loc[df['Escalate'], 'Priority'] = 'High'
    df.drop(columns=['Escalate'], inplace=True)
    return df
if __name__ == "__main__":
   df = pd.read_csv("temp_data.csv")
   df = modify_Priority(df)
   df.to_csv("temp_data.csv", index=False)