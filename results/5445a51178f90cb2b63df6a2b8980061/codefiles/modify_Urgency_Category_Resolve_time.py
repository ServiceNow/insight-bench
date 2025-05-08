import pandas as pd
from datetime import timedelta

def modify_Urgency_Category_Resolve_time(df: pd.DataFrame) -> pd.DataFrame:
    # Define the time window for the systemic issue (e.g., the burst lasts for 3 days starting on a specific date)
    update_start_date = pd.to_datetime('2023-10-01')
    update_end_date = update_start_date + timedelta(days=3)

    # Modify the DataFrame
    df_copy = df.copy()
    
    # Apply transformation: set 'Urgency' to "High" and increase 'Resolve time' for affected incidents
    mask = (df_copy['Opened'] >= update_start_date) & (df_copy['Opened'] < update_end_date)
    df_copy.loc[mask, 'Urgency'] = 'High'
    
    # Assuming we want to increase the 'Resolve time' by 50% during the burst
    df_copy.loc[mask, 'Resolve time'] *= 1.5

    return df_copy
if __name__ == "__main__":
   df = pd.read_csv("temp_data.csv")
   df = modify_Urgency_Category_Resolve_time(df)
   df.to_csv("temp_data.csv", index=False)