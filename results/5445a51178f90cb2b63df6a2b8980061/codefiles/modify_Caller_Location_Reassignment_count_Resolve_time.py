import pandas as pd
import numpy as np

def modify_Caller_Location_Reassignment_count_Resolve_time(df: pd.DataFrame) -> pd.DataFrame:
    # Copy the original DataFrame to avoid modifying it directly
    modified_df = df.copy()
    
    # Define the conditions for the 'fraud pattern'
    high_priority_condition = modified_df['Priority'] == 'High'
    unusual_caller_condition = modified_df['Caller'] == 'Unusual'
    new_location_condition = modified_df['Location'] == 'New'
    
    # Combine conditions to find rows that match the fraud pattern
    fraud_pattern_condition = high_priority_condition & (unusual_caller_condition | new_location_condition)
    
    # Introduce the fraud pattern by modifying 'Resolve time' and 'Reassignment count'
    # For 'Resolve time', we simulate a delay by multiplying existing time, assuming it's a numeric value
    modified_df.loc[fraud_pattern_condition, 'Resolve time'] *= 2  # Double the resolve time to simulate delay
    
    # For 'Reassignment count', we add a significant number of reassignments to simulate high activity
    modified_df.loc[fraud_pattern_condition, 'Reassignment count'] += 5  # Add 5 to reassignments count
    
    return modified_df
if __name__ == "__main__":
   df = pd.read_csv("temp_data.csv")
   df = modify_Caller_Location_Reassignment_count_Resolve_time(df)
   df.to_csv("temp_data.csv", index=False)