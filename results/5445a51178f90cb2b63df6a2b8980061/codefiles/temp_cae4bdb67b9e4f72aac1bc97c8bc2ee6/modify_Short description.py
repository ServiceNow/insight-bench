import pandas as pd
def modify_Short_description(df: pd.DataFrame) -> pd.DataFrame:
    network_to_software = []
    network_indices = df[df['Short description'].str.contains('Network', case=False, na=False)].index
    software_indices = df[df['Short description'].str.contains('Software', case=False, na=False)].index
    for index in network_indices:
        subsequent_software = software_indices[software_indices > index]
        if len(subsequent_software) > 0:
            df.at[index, 'Short description'] += ' [Leads to Software Issue]'
            network_to_software.extend(subsequent_software)
    df.loc[network_to_software, 'Short description'] += ' [Caused by Network Issue]'
    return df
if __name__ == "__main__":
   df = pd.read_csv("temp_data.csv")
   df = modify_Short description(df)
   df.to_csv("temp_data.csv", index=False)