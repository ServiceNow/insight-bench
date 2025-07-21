


# Granger Causality Pipeline



Required Imports



``` python
# Required Imports
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt
import pickle
```


Data Preparation


``` python
# Data Preparation
def load_data(filepath):
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    # Assuming the dataframe has a datetime index and columns for each cryptocurrency
    return df
```


Data Manipulation



``` python
# Data Manipulation
# Ensure the data is stationary if necessary
def check_stationarity(data, column):
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(data[column])
    print(f'ADF Statistic for {column}: {result[0]}')
    print('p-value:', result[1])
    if result[1] > 0.05:
        print(f"{column} is not stationary")
        return False
    else:
        print(f"{column} is stationary")
        return True
def make_stationary(data):
    return data.diff().dropna()
```



Granger Causality Implementation


``` python
# Model Implementation
def granger_test(data, variables, max_lags):
    results = {}
    for var in variables:
        result = grangercausalitytests(data[[variables[0], var]], max_lags, verbose=False)
        results[var] = result
    return results
```



Visualization



``` python
# Visualization
def plot_granger_results(results, variables, max_lags):
    fig, axes = plt.subplots(nrows=len(variables)-1, figsize=(10, 5 * (len(variables)-1)))
    if len(variables)-1 == 1:
        axes = [axes]  # Make it iterable
    for idx, var in enumerate(variables[1:]):
        for lag in range(1, max_lags+1):
            p_value = results[var][lag][0]['ssr_chi2test'][1]
            axes[idx].bar(lag, -np.log10(p_value))
        axes[idx].set_title(f'Granger Causality from {variables[0]} to {var}')
        axes[idx].set_xlabel('Lags')
        axes[idx].set_ylabel('-log10(p-value)')
    plt.tight_layout()
    plt.show()
```


Evaluate Results



``` python
# Evaluate Results
def evaluate_results(results, variables):
    for var in variables[1:]:
        for lag in results[var].keys():
            test_result = results[var][lag][0]['ssr_chi2test']
            print(f'Result for {var} at lag {lag}: Statistic = {test_result[0]}, p-value = {test_result[1]}')
```



Save Results



``` python
# Save Results
def save_results(results, filename):
    with open(filename, 'wb') as file:
        pickle.dump(results, file)
```


Main Execution



``` python
# Main Execution
def main():
    data = load_data('crypto_prices.csv')
    variables = ['BTC', 'ETH', 'LTC']  # Example cryptocurrencies
    if not check_stationarity(data, variables[0]):
        data = make_stationary(data)
    results = granger_test(data, variables, max_lags=5)
    plot_granger_results(results, variables, max_lags=5)
    evaluate_results(results, variables)
    save_results(results, 'granger_results.pkl')
if __name__ == '__main__':
    main()
```

