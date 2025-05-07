import pandas as pd
def modify_Category(df: pd.DataFrame) -> pd.DataFrame:
    network_indices = df[df['Category'] == 'Network'].index
    software_indices = df[df['Category'] == 'Software'].index
    for i in network_indices:
        next_software = software_indices[software_indices > i]
        if len(next_software) > 0:
            df.at[next_software[0], 'Category'] = 'Network->Software'
    return df
if __name__ == "__main__":
   df = pd.read_csv("temp_data.csv")
   df = modify_Category(df)
   df.to_csv("temp_data.csv", index=False)