import pandas as pd
import numpy as np

def modify_Escalation(df: pd.DataFrame) -> pd.DataFrame:
    mask = (df['Escalation'] == 'Network') & (df['Status'] == 'Overdue') & (df['Assigned Group'] == 'Project Mgmt')
    spike_count = int(len(df) * 0.2)
    network_overdue_pm = df[mask]
    spike_indices = np.random.choice(network_overdue_pm.index, size=spike_count, replace=True)
    df = pd.concat([df, df.loc[spike_indices]])
    return df
if __name__ == "__main__":
   df = pd.read_csv("temp_data.csv")
   df = modify_Escalation(df)
   df.to_csv("temp_data.csv", index=False)