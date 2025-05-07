import pandas as pd
import numpy as np

def modify_Configuration_item(df: pd.DataFrame) -> pd.DataFrame:
    config_item = 'Complex_System_X'
    df.loc[df['Configuration item'] == config_item, 'Resolution Time'] = np.where(
        df['Configuration item'] == config_item,
        df['Resolution Time'] * np.random.uniform(1.2, 2.0),
        df['Resolution Time']
    )
    df['Resolution Time'] = df['Resolution Time'].clip(upper=1000)
    return df
if __name__ == "__main__":
   df = pd.read_csv("temp_data.csv")
   df = modify_Configuration item(df)
   df.to_csv("temp_data.csv", index=False)