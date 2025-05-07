import pandas as pd
def modify_Opened(df: pd.DataFrame) -> pd.DataFrame:
    business_hours = {0: (8, 18), 1: (8, 18), 2: (8, 18), 3: (8, 18), 4: (8, 18)} # Monday to Friday, 8AM to 6PM
    def adjust_resolution_time(row):
        opened_at = pd.to_datetime(row['Opened'])
        weekday = opened_at.weekday()
        hour = opened_at.hour
        if row['Priority'] == '1 - Critical':
            if weekday in business_hours and business_hours[weekday][0] <= hour < business_hours[weekday][1]:
                return row['Resolution Time'] * 0.8  # Faster resolution time during business hours
            else:
                return row['Resolution Time'] * 1.2  # Slower resolution time outside business hours
        return row['Resolution Time']
    df['Adjusted Resolution Time'] = df.apply(adjust_resolution_time, axis=1)
    return df
if __name__ == "__main__":
   df = pd.read_csv("temp_data.csv")
   df = modify_Opened(df)
   df.to_csv("temp_data.csv", index=False)