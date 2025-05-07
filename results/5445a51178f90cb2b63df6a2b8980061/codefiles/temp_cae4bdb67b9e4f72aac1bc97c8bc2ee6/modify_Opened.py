import pandas as pd
def modify_Opened(df: pd.DataFrame) -> pd.DataFrame:
    if pd.api.types.is_datetime64_any_dtype(df['Opened']):
        df['Month'] = df['Opened'].dt.month
        df['YearEndPeak'] = df['Month'].apply(lambda x: 1 if x in [11, 12] else 0)
    return df
if __name__ == "__main__":
   df = pd.read_csv("temp_data.csv")
   df = modify_Opened(df)
   df.to_csv("temp_data.csv", index=False)