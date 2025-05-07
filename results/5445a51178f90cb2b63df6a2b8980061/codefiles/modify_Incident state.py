import pandas as pd
def modify_Incident_state(df: pd.DataFrame) -> pd.DataFrame:
    threshold = 50  # Example threshold for major system upgrade impact
    unresolved_count = df['Incident state'].value_counts().get('On Hold', 0)
    if unresolved_count > threshold:
        df.loc[df['Incident state'] == 'On Hold', 'Impact'] = 'High due to system upgrade'
    return df
if __name__ == "__main__":
   df = pd.read_csv("temp_data.csv")
   df = modify_Incident state(df)
   df.to_csv("temp_data.csv", index=False)