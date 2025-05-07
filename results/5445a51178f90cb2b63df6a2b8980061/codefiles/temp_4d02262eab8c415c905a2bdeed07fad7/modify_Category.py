import pandas as pd
def modify_Category(df: pd.DataFrame) -> pd.DataFrame:
    df['Priority'] = 'Normal'
    df.loc[df['Category'] == 'Software', 'Priority'] = 'New High-Impact'
    return df
if __name__ == "__main__":
   df = pd.read_csv("temp_data.csv")
   df = modify_Category(df)
   df.to_csv("temp_data.csv", index=False)