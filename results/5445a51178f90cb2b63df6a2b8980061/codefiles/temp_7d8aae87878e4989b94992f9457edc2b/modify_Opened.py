import pandas as pd
import numpy as np

def modify_Opened(df: pd.DataFrame) -> pd.DataFrame:
    def adjust_resolution_probability(opened_hour):
        return 0.8 if 0 <= opened_hour < 6 or 22 <= opened_hour < 24 else 0.5
    
    df['Opened'] = pd.to_datetime(df['Opened'])
    df['Resolution_Quick'] = df['Opened'].dt.hour.apply(adjust_resolution_probability)
    return df
if __name__ == "__main__":
   df = pd.read_csv("temp_data.csv")
   df = modify_Opened(df)
   df.to_csv("temp_data.csv", index=False)