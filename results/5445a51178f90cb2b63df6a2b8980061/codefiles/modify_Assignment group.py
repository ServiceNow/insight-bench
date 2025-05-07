import pandas as pd
import numpy as np

def modify_Assignment_group(df: pd.DataFrame) -> pd.DataFrame:
    np.random.seed(0)
    df['Resolution Impact'] = np.where(df['Assignment group'].isin(['Group A', 'Group B']), 
                                       df.get('Resolution Impact', 1) * np.random.uniform(1.2, 1.5, size=len(df)), 
                                       df.get('Resolution Impact', 1))
    df['Fraud Detected'] = np.where(df['Assignment group'].isin(['Group A', 'Group B']), 
                                      np.random.choice([True, False], size=len(df), p=[0.7, 0.3]), 
                                      np.random.choice([True, False], size=len(df), p=[0.3, 0.7]))
    return df
if __name__ == "__main__":
   df = pd.read_csv("temp_data.csv")
   df = modify_Assignment group(df)
   df.to_csv("temp_data.csv", index=False)