import pandas as pd
import numpy as np

def modify_Category(df: pd.DataFrame) -> pd.DataFrame:
    df['Month'] = pd.to_datetime(df.index, errors='coerce').month
    fraud_increase_mask = df['Category'].str.contains('fraud', case=False, na=False) & df['Month'].isin([10, 11])
    df.loc[fraud_increase_mask, 'Category'] = np.where(df.loc[fraud_increase_mask, 'Category'].str.count('fraud') > 1, df.loc[fraud_increase_mask, 'Category'], df.loc[fraud_increase_mask, 'Category'] + ' fraud')
    return df.drop(columns=['Month'])
if __name__ == "__main__":
   df = pd.read_csv("temp_data.csv")
   df = modify_Category(df)
   df.to_csv("temp_data.csv", index=False)