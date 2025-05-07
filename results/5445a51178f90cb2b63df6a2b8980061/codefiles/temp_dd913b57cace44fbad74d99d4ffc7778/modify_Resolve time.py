import pandas as pd
import numpy as np

def modify_Resolve_time(df: pd.DataFrame) -> pd.DataFrame:
    df['Reassignment count'] = df.get('Reassignment count', pd.Series([0]*len(df)))
    mask = df['Assigned Group'] == 'IT Securities'
    df.loc[mask, 'Reassignment count'] += np.random.randint(2, 5, size=mask.sum())
    return df
if __name__ == "__main__":
   df = pd.read_csv("temp_data.csv")
   df = modify_Resolve time(df)
   df.to_csv("temp_data.csv", index=False)