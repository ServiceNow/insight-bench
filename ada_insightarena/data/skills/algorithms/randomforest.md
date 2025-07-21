# RandomForest Implementation Template
    
## Required Imports
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
```
    
## Data Preparation
```python
def prepare_data(data_path):
    df = pd.read_csv(data_path)
    df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis="columns", inplace=True)
    
    categorical_col = []
    for column in df.columns:
        if df[column].dtype == object and len(df[column].unique()) <= 50:
            categorical_col.append(column)
            
    df['Attrition'] = df.Attrition.astype("category").cat.codes
    categorical_col.remove('Attrition')
    
    label = LabelEncoder()
    for column in categorical_col:
        df[column] = label.fit_transform(df[column])
        
    return df, categorical_col
```
    
## Data Manipulation
```python
def split_data(data, target):
    X = data.drop(target, axis=1)
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test
```
    
## RandomForest Implementation
```python
def implement_randomforest(X_train, y_train, X_test, y_test):
    rf_clf = RandomForestClassifier(n_estimators=100)
    rf_clf.fit(X_train, y_train)
    return rf_clf
```
    
## Evaluation
```python
def evaluate_model(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
        
    elif train==False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")
```
    
## Main Execution
```python
def main():
    df, categorical_cols = prepare_data('path/to/data.csv')
    X_train, X_test, y_train, y_test = split_data(df, 'Attrition')
    model = implement_randomforest(X_train, y_train, X_test, y_test)
    evaluate_model(model, X_train, y_train, X_test, y_test, train=True)
    evaluate_model(model, X_train, y_train, X_test, y_test, train=False)
    
if __name__ == '__main__':
    main()
```