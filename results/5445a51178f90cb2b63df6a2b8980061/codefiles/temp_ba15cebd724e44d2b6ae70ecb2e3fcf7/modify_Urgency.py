import pandas as pd
import numpy as np

def modify_Urgency(df: pd.DataFrame) -> pd.DataFrame:
    df['Urgency'] = np.where(df['Category'] == 'Network', 
                             np.where(df['Urgency'] < 4, df['Urgency'] + 1, df['Urgency']), 
                             df['Urgency'])
    return df
if __name__ == "__main__":
   df = pd.read_csv("temp_data.csv")
   df = modify_Urgency(df)
   df.to_csv("temp_data.csv", index=False)