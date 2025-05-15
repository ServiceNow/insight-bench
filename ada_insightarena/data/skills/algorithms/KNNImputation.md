
# KNN Imputation Pipeline


Required Imports



``` python
# Required Imports
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
```



Data Preparation



``` python
# Data Preparation
def load_data(filepath):
    return pd.read_csv(filepath)
```



Data Manipulation



``` python
# Data Manipulation
def select_features(data, features):
    return data[features]
```



KNN Imputation Implementation



``` python
# Model Implementation
def perform_knn_imputation(data, n_neighbors=5):
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_data = imputer.fit_transform(data)
    return pd.DataFrame(imputed_data, columns=data.columns)
```



Visualization



``` python
# Visualization
def plot_imputed_values(original_data, imputed_data, feature):
    plt.figure(figsize=(12, 6))
    plt.plot(original_data.index, original_data[feature], label='Original', marker='o')
    plt.plot(imputed_data.index, imputed_data[feature], label='Imputed', linestyle='--', marker='x')
    plt.title('Comparison of Original and Imputed Values')
    plt.xlabel('Index')
    plt.ylabel(feature)
    plt.legend()
    plt.show()
```



Evaluate Results



``` python
# Evaluate Results
def evaluate_imputation(original_data, imputed_data):
    mse = mean_squared_error(original_data, imputed_data)
    print(f"Mean Squared Error of imputation: {mse}")
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
    df = load_data('data_with_missing_values.csv')
    features = ['Feature1', 'Feature2', 'Feature3']  # Example features
    selected_data = select_features(df, features)
    imputed_data = perform_knn_imputation(selected_data)
    plot_imputed_values(df[features], imputed_data, 'Feature1')
    evaluate_imputation(df[features], imputed_data)
    save_results(imputed_data, 'imputed_data.csv')

if __name__ == '__main__':
    main()
```

