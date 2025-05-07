import pandas as pd
def modify_Escalation(df: pd.DataFrame) -> pd.DataFrame:
    if 'Escalation' not in df.columns or 'Category' not in df.columns or 'Date' not in df.columns:
        return df
    df['Date'] = pd.to_datetime(df['Date'])
    specific_month = '2023-05'  # Example specific month
    mask = (df['Date'].dt.strftime('%Y-%m') == specific_month) & (df['Category'] == 'Network')
    df.loc[mask, 'Escalation'] = df.loc[mask, 'Escalation'].apply(lambda x: 'High' if x != 'High' else x)
    return df
if __name__ == "__main__":
   df = pd.read_csv("temp_data.csv")
   df = modify_Escalation(df)
   df.to_csv("temp_data.csv", index=False)