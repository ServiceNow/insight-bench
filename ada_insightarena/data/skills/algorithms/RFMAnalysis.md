

# RFM Pipeline


Required Imports


``` python
# Required Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```


Data Preparation


``` python
# Data Preparation
def load_data(filepath):
    return pd.read_csv(filepath, parse_dates=['trans_date'])
```

Data Manipulation



``` python
# Data Manipulation
def calculate_rfm(data, analysis_date):
    data['hist'] = (analysis_date - data['trans_date']).dt.days
    recent_transactions = data[data['hist'] <= 730]  # Considering transactions in the last 2 years
    rfm_table = recent_transactions.groupby('customer_id').agg(
        recency=('hist', 'min'),
        frequency=('customer_id', 'size'),
        monetary_value=('tran_amount', 'sum')
    ).rename(columns={
        'hist': 'recency',
        'customer_id': 'frequency',
        'tran_amount': 'monetary_value'
    })
    return rfm_table
```



RFM Implementation



``` python
# Model Implementation
def assign_rfm_scores(data, quantiles):
    data['R_Quartile'] = pd.qcut(data['recency'], 4, labels=range(4, 0, -1))
    data['F_Quartile'] = pd.qcut(data['frequency'], 4, labels=range(1, 5))
    data['M_Quartile'] = pd.qcut(data['monetary_value'], 4, labels=range(1, 5))
    return data
```



Visualization



``` python
# Visualization
def plot_rfm_distribution(data):
    data['RFM_Score'] = data['R_Quartile'].astype(str) + data['F_Quartile'].astype(str) + data['M_Quartile'].astype(str)
    data['RFM_Score'].value_counts().sort_index().plot(kind='bar', color='skyblue')
    plt.title('RFM Score Distribution')
    plt.xlabel('RFM Score')
    plt.ylabel('Customer Count')
    plt.show()
```



Evaluate Results



``` python
# Evaluate Results
def analyze_rfm_scores(data):
    print(data.describe())
```

Save Results



``` python
# Save Results
def save_results(data, filename):
    data.to_csv(filename, index=False)
```

Main Execution



``` python
# Main Execution
def main():
    df = load_data('Retail_Data_Transactions.csv')
    analysis_date = pd.Timestamp('2015-04-01')
    rfm_table = calculate_rfm(df, analysis_date)
    quantiles = rfm_table.quantile(q=[0.25, 0.5, 0.75])
    rfm_scores = assign_rfm_scores(rfm_table, quantiles)
    plot_rfm_distribution(rfm_scores)
    analyze_rfm_scores(rfm_scores)
    save_results(rfm_scores, 'rfm_scores.csv')

if __name__ == '__main__':
    main()
```
