# What is A/B Testing?

**A/B testing** is a type of experiment in which you split your web traffic or user base into two groups, and show two different versions of a web page, app, email, and so on, with the goal of comparing the results to find the more successful version. With an A/B test, one element is changed between the original (a.k.a, “the control”) and the test version to see if this modification has any impact on user behavior or conversion rates.

**A/B testing** is an important element in decision making, especially when making business decisions. It is because often the assumed results are different from the facts that occur in the real world. When this happens, it can cause "disaster" in a company.

From a data scientist’s perspective, A/B testing is a form of statistical hypothesis testing or a significance test.

The steps to perform A/B testing are divided into 6 parts, namely:
1. Defining Goals
2. Identifying Metrics
3. Developing Hypothesis
4. Setting up Experiment
5. Running the Experiment
6. Analysing A/B Testing Results

# Defining Goals, Identifying Metrics, and Developing Hypothesis

For this project, we will run an A/B test to determine which is the better product ad pop-up layout between 2 designs. The 2 designs that will be tested are shown below.

![Screenshot 2022-08-24 164638.jpg](Ttest_files/41800914-c5f8-4690-b776-1976d56c843f.jpg)

From the image, we can see that Design A is the old design (control group) and Design B is the new design (treatment group).

Assuming that the marketing manager said that Design B will increase the click through rate (CTR) average compared to Design A. In this case, **CTR** is the **metrics** that we will analyze.

We can develop the **hypothesis**:

H0 : The average CTR of Design B is less than or equal to the average CTR of Design A, μB ≤ μA

Ha : The average CTR of Design B is more than the average CTR of Design A, μB > μA

# Setting Up Experiment

In the preparation of the experiment, there are 4 stages that must be carried out, which are determining the sample size, determining the duration of the A/B test, defining the control and treatment groups, and randomization.

You can read how to determine the sample size of an A/B test [here](https://towardsdatascience.com/required-sample-size-for-a-b-testing-6f6608dd330a) and how to determine the duration of an A/B test [here](https://www.abtasty.com/blog/how-long-run-ab-test/).

The **control group** is the **old feature** that you assume needs a change and the **treatment group** is the **new feature**. Randomization means that the control group and the treatment group are tested in a randomly distributed segment.

# Running the Experiment

The experiment ran for 10 days, from November 13, 2021 to November 22, 2021 and spread to 59,984 different users. The experiment result is stored in a CSV file to be analyzed later.

Now that we have the A/B test result, let's begin to process it. Like usual, let's start by importing some necessary libraries and read the dataset.


```python
pip install pingouin
```

   


```python
import numpy as np
import pandas as pd
import pingouin as pg
from statsmodels.stats.weightstats import ttost_paired
from scipy import stats
```


```python
data = pd.read_csv('../input/ctr-a-b/ctr_a_b.csv')
data
```

```python
data.info()
```


```python
data['ctr'] = data['ctr'].dropna()
data['dt'] = pd.to_datetime(data['dt'])
data['groupid'] = data['groupid'].replace([0, 1], ['a', 'b'])
data = data.rename(columns = {'groupid':'design'})

data.info()
```

Now that all the datatypes are correct, let's check the unique values for every columns.


```python
data['dt'].unique()
```

```python
data['design'].unique()
```





```python
len(data['userid'].unique())
```


```python
group = data.groupby(['dt','design']).mean('ctr')
group
```


# T-Test

```python
design_a = group.query('design == "a"')['ctr']
design_b = group.query('design == "b"')['ctr']
```

# SciPy T-Test


```python
sci = stats.ttest_rel(design_a, design_b)
sci
```

# Pingouin T-Test


```python
stats.levene(design_a, design_b)
```

```python
ping = pg.ttest(design_a, design_b, paired = True, correction = False)
ping
```
