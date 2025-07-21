

# Prophet Implementation Pipeline


Required Imports

``` python
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.offline as py
py.init_notebook_mode()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet.plot import add_changepoints_to_plot
%matplotlib inline
plt.style.use('fivethirtyeight')
```


Data Preparation


``` python
def prepare_data(data_path):
    data = pd.read_csv(data_path)
    return data
```


Data Manipulation

``` python
# Function for feature engineering and data transformation
def transform_data(data):
  data = data.rename(columns={data.columns[0]: 'ds',data.columns[1]: 'y'})
  return data
```



Prophet Implementation



``` python
# Model class or function implementation
def implement_prophet(processed_data, interval_width=0.8):
    my_model = Prophet(interval_width)
    model_fit=my_model.fit(processed_data)
    return model_fit
```



Visualization



``` python
# Visualization functions
def plot_results(model_fit, processed_data):
    residuals = pd.DataFrame(model_fit.resid)
    fig, ax = plt.subplots(1,2)
    residuals.plot(title="Residuals", ax=ax[0])
    residuals.plot(kind='kde', title='Density', ax=ax[1])
    plt.show()

    # Actual vs Fitted
    model_fit.plot_predict(dynamic=False)
    plt.show()
```



Evaluate Results



``` python
def evaluate_model(model, test):
    # Compute forecasting and plot
    fore_cast = model.predict(len(test), alpha=0.05)
    fc_series = pd.Series(fore_cast, index=test.index)
    model.plot(fore_cast, uncertainty=True)
    model.plot_components(forecast)
    plt.plot(test, label='actual')
    plt.plot(fc_series, label='forecast')
    # plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)
    plt.title('Forecast vs Actuals')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()
```



Adjusting Trend & Adding Changepoints to Prophet



``` python
def trend_changepoint(processed_data,test):
  pro_change= Prophet(n_changepoints=20, yearly_seasonality=True, changepoint_prior_scale=0.08)
  forecast = pro_change.fit(processed_data).predict(test)
  fig= pro_change.plot(forecast);
  a = add_changepoints_to_plot(fig.gca(), pro_change, forecast)
```



Save Results



``` python
# Save model results
def save_model_results(model_fit, output_path):
    # Here, it might be a pickled model, a CSV of coefficients, etc.
    # We'll pretend its a pickled model
    with open(output_path, 'wb') as f:
        pickle.dump(model_fit, f)
```



Main Execution



``` python
def main():
    # Tie everything together
    data = prepare_data('path/to/data')
    # Here we assume that we need just a single column named 'value'. Otherwise, data selection should be done
    processed_data = transform_data(data)
    model_fit = implement_prophet(processed_data)
    plot_results(model_fit, processed_data)
    trend_changepoint(processed_data,test)

    # Create train and test data
    train = processed_data.iloc[:int(0.7 * len(processed_data))]
    test = processed_data.iloc[int(0.7 * len(processed_data)):]
    evaluate_model(model_fit, test)

    save_model_results(model_fit, 'output/path')

if __name__ == '__main__':
    main()
```

