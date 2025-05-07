import pandas as pd
import numpy as np

def modify_Category(df: pd.DataFrame) -> pd.DataFrame:
    if 'Category' in df.columns and 'Incidents' in df.columns:
        network_outage = df['Category'] == 'Network'
        if network_outage.any():
            df.loc[network_outage, 'Incidents'] *= 1.5
            df['Urgency'] = np.where(network_outage, df.get('Urgency', pd.Series(np.nan, index=df.index)) + 1, df.get('Urgency', pd.Series(np.nan, index=df.index)))
            df['Impact'] = np.where(network_outage, df.get('Impact', pd.Series(np.nan, index=df.index)) + 1, df.get('Impact', pd.Series(np.nan, index=df.index)))
    return df
if __name__ == "__main__":
   df = pd.read_csv("temp_data.csv")
   df = modify_Category(df)
   df.to_csv("temp_data.csv", index=False)