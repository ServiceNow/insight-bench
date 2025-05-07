import pandas as pd
def modify_Closed(df: pd.DataFrame) -> pd.DataFrame:
    if 'Impact' in df.columns and 'Priority' in df.columns:
        high_impact_categories = df['Impact'].unique()  # Assuming all categories need checking
        df['Prior_Impact'] = df['Impact'].shift()
        df['Priority'] = df.apply(
            lambda row: 'High' if row['Prior_Impact'] in high_impact_categories and row['Priority'] == 'Low' else row['Priority'], axis=1)
        df.drop(columns=['Prior_Impact'], inplace=True)
    return df
if __name__ == "__main__":
   df = pd.read_csv("temp_data.csv")
   df = modify_Closed(df)
   df.to_csv("temp_data.csv", index=False)