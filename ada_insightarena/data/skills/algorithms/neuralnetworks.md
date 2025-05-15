# NeuralNetworks Implementation Template
    
## Required Imports
```python
# All necessary imports for the entire implementation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
import warnings
warnings.filterwarnings("ignore") #supresses warnings
```
    
## Data Preparation
```python
# Function for data cleaning and preprocessing
def prepare_data(data_url):
    full_df = pd.read_csv(data_url)
    full_df.drop(['Unnamed: 0'], inplace=True, axis=1)
    X_train = full_df.drop('target', inplace=False, axis=1) #remove 'target' column from input features
    y_train = full_df['target'] #stores target (1 or 0) in a separate array
    X_train = X_train.reset_index(drop=True) 
    y_train = y_train.reset_index(drop=True)
    X_train = np.array(X_train, dtype=float)
    y_train = np.array(y_train, dtype=float).reshape(-1,1)
    
    return X_train, y_train
```
    
## NeuralNetworks Implementation
```python
# Model class or function implementation
class Perceptron:
    def __init__(self, x, y):
        self.input = np.array(x, dtype=float) 
        self.label = np.array(y, dtype=float)
        self.weights = np.random.rand(x.shape[1], y.shape[1]) #randomly initialize the weights
        self.z = self.input @ self.weights #dot product of the vectors
        self.yhat = self.sigmoid(self.z) #apply activation function

    def sigmoid(self, x):
        return 1.0/(1.0+np.exp(-x))

    def sigmoid_deriv(self, x):
        s = sigmoid(x)
        return s * (1 - s)

    def forward_prop(self):
        self.yhat = self.sigmoid(self.input @ self.weights)
        return self.yhat

    def back_prop(self):
        gradient = self.input.T @ (-2.0 * (self.label - self.yhat) * self.sigmoid(self.yhat))
        self.weights -= gradient #process of finding the minimum loss
```
    
## Visualization
```python
# Visualization functions
def plot_results(history):
    # Generic plotting code for NeuralNetworks results
    plt.plot(history)
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Training Iteration')
    plt.show()
```
    
## Save Results
```python
# TO DO: Save model and results
```
    
## Main Execution
```python
def main():
    # Tie everything together
    X_train, y_train = prepare_data('https://raw.githubusercontent.com/karthikb19/data/master/breastcancer.csv')
    model = Perceptron(X_train, y_train)

    # TO DO: Modify training to not require static epochs 
    training_iterations = 1000
    history = [] 
    for i in range(training_iterations):
        model.forward_prop()
        model.back_prop()
        yhat = model.forward_prop()
        history.append(np.mean((yhat - model.label)**2)) #mean squared error

    plot_results(history)

if __name__ == '__main__':
    main()
```