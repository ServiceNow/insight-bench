## Feature Selection
#### Feature selection is the process of reducing the number of input variables when developing a predictive model. It is desirable to reduce the number of input variables to both reduce the computational cost of modeling and, in some cases, to improve the performance of the model.

## Pearson Correlation
_______
#### Pearson's correlation coefficient is the test statistics that measures the statistical relationship, or association, between two continuous variables. It gives information about the magnitude of the association, or correlation, as well as the direction of the relationship.
#### There are two types of correlations. Positive Correlation: means that if feature A increases then feature B also increases or if feature A decreases then feature B also decreases. Both features move in tandem and they have a linear relationship. Negative Correlation: means that if feature A increases then feature B decreases and vice versa.

![](https://www.researchgate.net/profile/Ivan-Nikolov/publication/345362331/figure/fig11/AS:954798429978645@1604653093872/Correlation-matrix-of-the-used-metrics-together-with-the-dependent-variable-For-easier.ppm)

#### We will be choosing our features after calculations based on correlation matrix.
#### If 2 or more independent features are highly correlated then they can be considered as duplicate features and can be dropped. When independent variables are highly correlated, change in one variable would cause change to another and so the model results fluctuate significantly. The model results will be unstable and vary a lot given a small change in the data or model. Both positive and negative correlations are taken into consideration.  
     
_______

### Importing required libraries


```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline 
import warnings
warnings.filterwarnings("ignore")


from sklearn.model_selection import train_test_split
def split(df,label):
    X_tr, X_te, Y_tr, Y_te = train_test_split(df, label, test_size=0.25, random_state=42)
    return X_tr, X_te, Y_tr, Y_te


def correlation(dataset, cor):
    df = dataset.copy()
    col_corr = set()  # For storing unique value
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > cor: # absolute values to handle positive and negative correlations
                colname = corr_matrix.columns[i]  
                col_corr.add(colname)
    df.drop(col_corr,axis = 1,inplace = True)
    return df


from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score

classifiers = ['LinearSVM', 'RadialSVM', 
               'Logistic',  'RandomForest', 
               'AdaBoost',  'DecisionTree', 
               'KNeighbors','GradientBoosting']

models = [svm.SVC(kernel='linear'),
          svm.SVC(kernel='rbf'),
          LogisticRegression(max_iter = 1000),
          RandomForestClassifier(n_estimators=200, random_state=0),
          AdaBoostClassifier(random_state = 0),
          DecisionTreeClassifier(random_state=0),
          KNeighborsClassifier(),
          GradientBoostingClassifier(random_state=0)]


def acc_score(df,label):
    Score = pd.DataFrame({"Classifier":classifiers})
    j = 0
    acc = []
    X_train,X_test,Y_train,Y_test = split(df,label)
    for i in models:
        model = i
        model.fit(X_train,Y_train)
        predictions = model.predict(X_test)
        acc.append(accuracy_score(Y_test,predictions))
        j = j+1     
    Score["Accuracy"] = acc
    Score.sort_values(by="Accuracy", ascending=False,inplace = True)
    Score.reset_index(drop=True, inplace=True)
    return Score


def acc_score_cor(df,label,cor_list):
    Score = pd.DataFrame({"Classifier":classifiers})
    for k in range(len(cor_list)):
        df2 = correlation(df, cor_list[k])
        X_train,X_test,Y_train,Y_test = split(df2,label)
        j = 0
        acc = []
        for i in models:
            model = i
            model.fit(X_train,Y_train)
            predictions = model.predict(X_test)
            acc.append(accuracy_score(Y_test,predictions))
            j = j+1  
        feat = str(cor_list[k])
        Score[feat] = acc
    return Score

        
def plot2(df,l1,l2,p1,p2,c = "b"):
    feat = df.columns.tolist()
    feat = feat[1:]
    plt.figure(figsize = (16, 18))
    for j in range(0,df.shape[0]):
        value = []
        k = 0
        for i in range(1,len(df.columns.tolist())):
            value.append(df.iloc[j][i])
        plt.subplot(4, 4,j+1)
        ax = sns.pointplot(x=feat, y=value,color = c )
        plt.text(p1,p2,df.iloc[j][0])
        plt.xticks(rotation=90)
        ax.set(ylim=(l1,l2))
        k = k+1
        

def highlight_max(data, color='aquamarine'):
    attr = 'background-color: {}'.format(color)
    if data.ndim == 1:  
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else: 
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)
```

________
### Function Description

#### 1. split():
Splits the dataset into training and test set.

#### 2. correlation():
Returns the dataframe after dropping features with greater correlation than the given value.

#### 3. acc_score():
Returns accuracy for all the classifiers.

#### 4. acc_score_cor():
Returns accuracy for all the classifiers after dropping features for the respective correlation value.

#### 5. plot2():
For plotting the results.
_____
### The following 3 datasets are used:
1. Breast Cancer
2. Parkinson's Disease
3. PCOS
______
### Plan of action:
* Looking at dataset (includes a little preprocessing)
* Heatmap (Plotting the heatmap)
* Checking Accuracy (comparing accuracies with the new dataset)
* Visualization (Plotting the graphs)
_______

__________
# Breast Cancer
_________________

### 1. Looking at dataset


```python
data_bc = pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")
label_bc = data_bc["diagnosis"]
label_bc = np.where(label_bc == 'M',1,0)
data_bc.drop(["id","diagnosis","Unnamed: 32"],axis = 1,inplace = True)

print("Breast Cancer dataset:\n",data_bc.shape[0],"Records\n",data_bc.shape[1],"Features")
```



```python
display(data_bc.head())
print("All the features in this dataset have continuous values")
```

    All the features in this dataset have continuous values


### 2. Heatmap


```python
plt.figure(figsize=(18,18))
cor1 = data_bc.corr()
sns.heatmap(cor1, annot=True, cmap="viridis",annot_kws={"size":8})
plt.show()
```
    


### 3. Checking Accuracy


```python
score1 = acc_score(data_bc,label_bc)
score1
```

```python
corrate_bc = [0.6,0.7,0.8,0.9,0.95,0.99]
classifiers = score1["Classifier"].tolist()
score_bc = acc_score_cor(data_bc,label_bc,corrate_bc)
score_bc.style.apply(highlight_max, subset = score_bc.columns[1:], axis=None)
```

#### Best Accuracy with all features : RandomForest Classifier - 0.972
#### Best Accuracy after applying with correlation() : LinearSVM - for = (0.9,0.99) - 0.972  and DecisionTree Classifier - for = (0.9) - 0.972
#### Here we see no improvement.

### 4. Visualization


```python
plot2(score_bc,0.85,1,2.4,0.86,c = "gold")
```
  


________
# Parkinson's Disease
___________

### 1. Looking at dataset


```python
data_pd = pd.read_csv("../input/parkinson-disease-detection/Parkinsson disease.csv")
label_pd = data_pd["status"]
data_pd.drop(["status","name"],axis = 1,inplace = True)

print("Parkinson's disease dataset:\n",data_pd.shape[0],"Records\n",data_pd.shape[1],"Features")
```

```python
display(data_pd.head())
print("All the features in this dataset have continuous values")
```


### 2. Heatmap


```python
plt.figure(figsize=(18,18))
cor3 = data_pd.corr()
sns.heatmap(cor3, annot=True, cmap="seismic",annot_kws={"size":8})
plt.show()
```

### 3. Checking Accuracy


```python
score3 = acc_score(data_pd,label_pd)
score3
```

________
# PCOS
________

### 1. Looking at dataset


```python
data_pcos = pd.read_csv("../input/pcos-dataset/PCOS_data.csv")
label_pcos = data_pcos["PCOS (Y/N)"]
data_pcos.drop(["Sl. No","Patient File No.","PCOS (Y/N)","Unnamed: 44","II    beta-HCG(mIU/mL)","AMH(ng/mL)"],axis = 1,inplace = True)
data_pcos["Marraige Status (Yrs)"].fillna(data_pcos['Marraige Status (Yrs)'].describe().loc[['50%']][0], inplace = True) 
data_pcos["Fast food (Y/N)"].fillna(1, inplace = True) 

print("PCOS dataset:\n",data_pcos.shape[0],"Records\n",data_pcos.shape[1],"Features")
```


```python
display(data_pcos.head())
print("The features in this dataset have both discrete and continuous values")
```

    The features in this dataset have both discrete and continuous values


### 2. Heatmap


```python
plt.figure(figsize=(18,18))
cor4 = data_pcos.corr()
sns.heatmap(cor4, annot=True, cmap="YlOrBr",annot_kws={"size":8})
plt.show()
```

### 3. Checking Accuracy


```python
score4 = acc_score(data_pcos,label_pcos)
score4
```

```python
corrate_pcos = [0.4,0.5,0.6,0.8,0.9,0.95]
classifiers = score4["Classifier"].tolist()
score_pcos = acc_score_cor(data_pcos,label_pcos,corrate_pcos)
score_pcos.style.apply(highlight_max, subset = score_pcos.columns[1:], axis=None)
```

#### Best Accuracy with all features : RandomForest Classifier - 0.889
#### Best Accuracy after applying with correlation() : DecisionTree Classifier - for = (0.9) - 0.904
#### Here we can see an improvement of ~1.5%.

### 4. Visualization


```python
plot2(score_pcos,0.60,1,2.5,0.975,c = "limegreen")
```

__________


#### From looking at these results we can see that there is a possibility of slight improvement in the accuracy after removing features that are correlated.
#### Link to other feature selection methods:
##### [Genetic Algorithm](https://www.kaggle.com/tanmayunhale/genetic-algorithm-for-feature-selection)
##### [Variance Threshold](https://www.kaggle.com/tanmayunhale/feature-selection-variance-threshold)
##### [F-score](https://www.kaggle.com/tanmayunhale/feature-selection-f-score)


```python

```
