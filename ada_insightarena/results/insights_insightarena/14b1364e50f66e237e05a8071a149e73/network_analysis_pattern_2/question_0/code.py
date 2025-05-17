import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Assume df is already loaded as per the input details
# Data Preparation & Cleaning
try:
    # Convert 'Reassignment count' and 'Resolve time' to numeric, handling missing values
    df['Reassignment count'] = pd.to_numeric(df['Reassignment count'], errors='coerce')
    df['Resolve time'] = pd.to_numeric(df['Resolve time'], errors='coerce')

    # Filter for high-priority incidents
    high_priority_df = df[df['Priority'] == '1 - High']

    # Drop rows with NaN values in 'Reassignment count' or 'Resolve time'
    high_priority_df = high_priority_df.dropna(subset=['Reassignment count', 'Resolve time'])

    # Check if the data is stationary using Augmented Dickey-Fuller test
    def check_stationarity(series):
        result = adfuller(series)
        return result[1] <= 0.05  # p-value <= 0.05 indicates stationarity

    # Make data stationary if needed
    if not check_stationarity(high_priority_df['Reassignment count']):
        high_priority_df['Reassignment count'] = high_priority_df['Reassignment count'].diff().dropna()

    if not check_stationarity(high_priority_df['Resolve time']):
        high_priority_df['Resolve time'] = high_priority_df['Resolve time'].diff().dropna()

    # Ensure no NaN values after differencing
    high_priority_df.dropna(subset=['Reassignment count', 'Resolve time'], inplace=True)

    # Check if there is enough data for Granger causality test
    if len(high_priority_df) > 5:  # Ensure there are more data points than the max lag
        # Granger Causality Test
        max_lag = 5
        granger_results = grangercausalitytests(high_priority_df[['Resolve time', 'Reassignment count']], max_lag, verbose=False)

        # Extract p-values for each lag
        p_values = [granger_results[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)]

        # Visualization
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, max_lag + 1), -np.log10(p_values), color='skyblue')
        plt.title('Granger Causality Test: Reassignment Count -> Resolve Time')
        plt.xlabel('Lag')
        plt.ylabel('-log10(p-value)')
        plt.xticks(range(1, max_lag + 1))
        plt.tight_layout()

        # Save the plot
        plt.savefig('insightarena/results/insights_insightarena/14b1364e50f66e237e05a8071a149e73/network_analysis_pattern_2/question_0/plot.jpeg')

        # Compute & Store Key Statistics
        stats = {
            'p_values': p_values,
            'min_p_value': min(p_values),
            'max_p_value': max(p_values),
            'stationarity_reassignment_count': check_stationarity(high_priority_df['Reassignment count']),
            'stationarity_resolve_time': check_stationarity(high_priority_df['Resolve time']),
        }

        # Print the stats dictionary
        print(stats)
    else:
        raise RuntimeError("Not enough data after preprocessing for Granger causality test.")

except Exception as e:
    raise RuntimeError(f"An error occurred during the analysis: {e}")
