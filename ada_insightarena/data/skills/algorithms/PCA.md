


# PCA Implementation Pipeline



Required Imports



``` python
# Required Imports
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
```



Data Preparation



``` python
# Data Preparation
def load_data(filepath):
    df = pd.read_csv(filepath)
    X = df.iloc[:, :-1]
    y = pd.Series(LabelEncoder().fit_transform(df.iloc[:, -1]))
    return X, y
```



Data Manipulation



``` python
# Data Manipulation
def split_data(X, y):
    return train_test_split(X, y, test_size=0.3, random_state=0)
```



PCA Implementation



``` python
class SVDPCA:
    def __init__(self, n_components=None):
        self.n_components = n_components

    @staticmethod
    def svd_flip_vector(U):
        max_abs_cols_U = np.argmax(np.abs(U), axis=0)
        # extract the signs of the max absolute values
        signs_U = np.sign(U[max_abs_cols_U, range(U.shape[1])])

        return U * signs_U

    def fit_transform(self, X):
        n_samples, n_features = X.shape
        X_centered = X - X.mean(axis=0)

        if self.n_components is None:
            self.n_components = min(n_samples, n_features)

        U, S, Vt = np.linalg.svd(X_centered)
        # flip the eigenvector sign to enforce deterministic output
        U_flipped = self.svd_flip_vector(U)

        self.explained_variance = (S[:self.n_components] ** 2) / (n_samples - 1)
        self.explained_variance_ratio = self.explained_variance / np.sum(self.explained_variance)

        # X_new = X * V = U * S * Vt * V = U * S
        X_transformed = U_flipped[:, : self.n_components] * S[: self.n_components]

        return X_transformed
```



Visualization



``` python
# Visualization
def plot_pca(components):
    plt.figure(figsize=(8, 6))
    plt.scatter(components[:, 0], components[:, 1], alpha=0.5)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Result')
    plt.grid(True)
    plt.show()
```



Evaluate Results



``` python
# Evaluate Results
def evaluate_pca(model):
    print('Explained variance:', model.explained_variance_)
    print('Explained variance ratio:', model.explained_variance_ratio_)
```



Save Results



``` python
# Save Results
def save_results(model, filename='pca_model.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
```



Main Execution



``` python
# Main Execution
def main():
    X, y = load_data('iris.csv')
    X_train, X_test, y_train, y_test = split_data(X, y)
    pca = SVDPCA(n_components=2)
    X_transformed = pca.fit_transform(X_train)
    plot_pca(X_transformed)
    evaluate_pca(pca)
    save_results(pca)
if __name__ == '__main__':
    main()
```
