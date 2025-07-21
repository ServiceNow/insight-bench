```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation


```

### Load Data 


```python
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>145</th>
      <td>6.7</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.3</td>
    </tr>
    <tr>
      <th>146</th>
      <td>6.3</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>1.9</td>
    </tr>
    <tr>
      <th>147</th>
      <td>6.5</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>148</th>
      <td>6.2</td>
      <td>3.4</td>
      <td>5.4</td>
      <td>2.3</td>
    </tr>
    <tr>
      <th>149</th>
      <td>5.9</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>1.8</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 4 columns</p>
</div>




```python
X = df.iloc[:, [0, 1, 2, 3]].values
X=X.astype(float)
X=np.array(X)
```

## PCA IMPLEMENTATION FROM SCRATCH

### IMPLEMENTATION OF EIGEN DECOMPOSITION FROM SCRATCH


```python
def power_iteration(matrix, num_iterations, tolerance):
    # Initialize a random vector as an initial guess for the eigenvector
    n = matrix.shape[0]
    eigenvector = np.random.rand(n)
    for _ in range(num_iterations):
        # Perform matrix-vector multiplication
        matrix_times_vector = np.dot(matrix, eigenvector)

        # Compute the eigenvalue as the Rayleigh quotient
        eigenvalue = np.dot(eigenvector, matrix_times_vector)

        # Normalize the eigenvector
        eigenvector = matrix_times_vector / np.linalg.norm(matrix_times_vector)

        # Check for convergence
        if np.linalg.norm(matrix_times_vector - eigenvalue * eigenvector) < tolerance:
            break

    return eigenvalue, eigenvector

def eigen(matrix, num_iterations=1000000, tolerance=1e-12):
    # Initialize arrays to store eigenvalues and eigenvectors
    eigenvalues = []
    eigenvectors = []
    matrix = np.array(matrix).astype(float)

    # Perform power iteration for each eigenvalue
    for _ in range(matrix.shape[0]):
        eigenvalue, eigenvector = power_iteration(matrix, num_iterations, tolerance)
        eigenvalues.append(eigenvalue)
        eigenvectors.append(eigenvector)

        # Deflate the matrix by removing the contribution of the found eigenvalue
        matrix -= eigenvalue * np.outer(eigenvector, eigenvector)

    return np.array(eigenvalues), np.array(eigenvectors)
```


```python
def _PCA(X, n_components=2):
    if n_components > X.shape[1]:
        raise ValueError("n_components cannot be greater than the number of features")

    # 1. Standardize data by subtracting mean and dividing by standard deviation
    mean_values = X.mean(axis=0)
    std_values = X.std(axis=0)
    X_standardized = (X - mean_values) / std_values

    # 2. Covariance matrix From Scratch
    cov = np.dot(X_standardized.T, X_standardized) / (X_standardized.shape[0]-1)  # (n_features, n_features)

    # 3. Eigen decomposition
    eigen_values, eigen_vectors = eigen(cov)

    # 4. Sort eigen values in descending order
    idx = eigen_values.argsort()[::-1]
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]

    # 5. Select top n_components
    eigen_vectors = eigen_vectors[:, :n_components]

    # 6. Project original data
    X_pca = np.dot(X_standardized, eigen_vectors)

    # 7. Inverse transform to get reconstructed data
    X_reconstructed = np.dot(X_pca, eigen_vectors.T) * std_values + mean_values

    return eigen_values, eigen_vectors, X_pca, X_reconstructed

```


```python
def loss (X, X_reconstructed):
    return (np.sum(abs(X - X_reconstructed))/X.shape[0])
```


```python
lossList=[]
for _ in range(1,X.shape[1]+1):
    eigen_values, eigen_vectors, X_pca, X_reconstructed = _PCA(X, n_components=_)
    lossList.append(loss(X,X_reconstructed).round(3))
    print("Loss for ",_, "components: ",loss(X,X_reconstructed).round(3))
```

    Loss for  1 components:  2.611
    
    Loss for  2 components:  2.378
    
    Loss for  3 components:  1.138
    
    Loss for  4 components:  0.0



```python
# Plotting the Loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, X.shape[1]+1), lossList, marker='o', color='green')
plt.xlabel('Number of components')
plt.ylabel('Loss')
plt.title('Loss vs. Number of components')
plt.show()
```


    
![png](data/skills/algorithms/EigenDecomposition_files/data/skills/algorithms/EigenDecomposition_10_0.png)
    



```python
eigen_values, eigen_vectors, X_pca, X_reconstructed = _PCA(X, n_components=2)
```

#### We will chosse the number of components to be 2 for the sake of visualization 


```python
# Plotting the X_pca 
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c="green", alpha=0.6, s=60, label="PCA", marker='o')
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')
plt.title('Principal Components')
plt.show()
```


    
![png](data/skills/algorithms/EigenDecomposition_files/data/skills/algorithms/EigenDecomposition_13_0.png)
    

