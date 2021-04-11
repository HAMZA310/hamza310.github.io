---
layout: page
title: feature
permalink: /feature-engieering/
---

```python
import pandas as pd
```

### Feature Engineering


Machine learning (ML), at its core, is a data driven way to write program. In other words, attempting to manually construct a program that produces predictions (e.g. classify whether an image is of a dog), we collect a data set that consists of a large number of images and their corresponding labels, then use a machine learning algorithm to automatically learn a kind of “program” that can produce these correct classifications on the data set that we have. 
Inputs, or [`features`](https://en.wikipedia.org/wiki/Feature_(machine_learning), are the information fed into the machine learning algorithm. While outputs in machine learning algorithms can be of different types (discrete, continuous, multivariate), we typically represent the input as a real-valued vector. 

In real world, datasets may not contain informative, discriminating and independent features which are crucial for effective algorithms. In this tutorial, we'll learn about extracting useful features from raw data- called `feature engineering`- the first step in the pipeline below. In practice, 'preparing data' could be very time consuming step- however, here, we'll discuss simple examples to illustrate the point. 

In the image below, observe a diagram showing a full pipleline of a general [Machine Learning](https://en.wikipedia.org/wiki/Machine_learning) Model. 

![pipe.png](attachment:pipe.png)



### `Feature Transformation`
Go to this [link](https://playground.tensorflow.org/#activation=sigmoid&batchSize=9&dataset=gauss&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=15&networkShape=1&seed=0.10855&showTestData=false&discretize=false&percTrainData=30&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false) and hit the play/run button at the top left. Image attached below for reference. Ignore the hidden layer or anything else which looks unfamiliar. We'll come back to hidden layers in the last quarter of this course.

![fig1.png](attachment:fig1.png)

The problem above is to classify whether input image is of +ve class or -ve class. You have two features x1 and x2, say length and width. When you hit run, it'll easily classify and training loss will approach zero. Now, let's consider a more difficult problem in which +ve and -ve classes are not easily separable on this [link](https://playground.tensorflow.org/#activation=sigmoid&batchSize=18&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=1&seed=0.28161&showTestData=false&discretize=false&percTrainData=30&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false). Image below.
![Screen%20Shot%202020-07-30%20at%203.54.19%20PM.png](attachment:Screen%20Shot%202020-07-30%20at%203.54.19%20PM.png). It turns out, using the previous set of features, it's not possible to classify this new dataset with sufficient accuracy- assumming everything else is kept same. Let's use $x_1^2 \ and \  x_2^2$ as features as well. So, 4 features in total. In this case, a circular decison boundary will classify the classes with good accuracy. Image below [(link)](https://playground.tensorflow.org/#activation=sigmoid&batchSize=18&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=1&seed=0.28161&showTestData=false&discretize=false&percTrainData=30&x=true&y=true&xTimesY=false&xSquared=true&ySquared=true&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false). 

![fig3.png](attachment:fig3.png)
We originally had only length and width. Now, we created new features by taking squares of each. This is called `Features Transformation`. Specifically, we created new features using the existing features. These new features may not have the same interpretation as the original features, but they have more discriminatory power in a different space than the original space- thereby creating a circular decision boundary.

### Exercise:
Observe these set of features on this [link](https://playground.tensorflow.org/#activation=sigmoid&batchSize=9&dataset=gauss&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=15&networkShape=1&seed=0.10855&showTestData=false&discretize=false&percTrainData=30&x=false&y=false&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=true&cosY=false&sinY=true&collectStats=false&problem=classification&initZero=false&hideText=false). Do these adequately classify? What if you increase the noise? Does the decision boudary look familiar?

#### `WARNING: ANSWER IN THE NEXT CELL. TO GAIN MAXIMUM BENEFIT PROCEED ONLY IF YOU'VE SPENT SUFFICIENT TIME ON THIIS EXERCISE `


```python

```


### Answer:
Yes, the accuracy is quite good as long as the noise is not too much. Since our features our sin(x1) and sin(x2). Decision boudaries look like [sin curve](https://www.google.com/search?q=sin+curve&oq=sin+curve&aqs=chrome..69i57j0l7.3064j0j7&sourceid=chrome&ie=UTF-8).


```python

```

### [Feature Scaliing](https://en.wikipedia.org/wiki/Feature_scaling):
Many algorithms such as SVM and KNN (more on both of them in future) expect their features of scaled. For example, if one of the features ranges from 0 to 1 and another from 500-10000. Then their performance will decline. Instead of scaling these features using native Python, we'll make use of open-source Machine Learning libraries in which people have already written these functions. One such library is [sklearn](https://scikit-learn.org/stable/).  



```python
from sklearn.datasets import load_linnerud
```


```python
linnerud_dict = load_linnerud()
data = linnerud_dict.data
data
```




    array([[  5., 162.,  60.],
           [  2., 110.,  60.],
           [ 12., 101., 101.],
           [ 12., 105.,  37.],
           [ 13., 155.,  58.],
           [  4., 101.,  42.],
           [  8., 101.,  38.],
           [  6., 125.,  40.],
           [ 15., 200.,  40.],
           [ 17., 251., 250.],
           [ 17., 120.,  38.],
           [ 13., 210., 115.],
           [ 14., 215., 105.],
           [  1.,  50.,  50.],
           [  6.,  70.,  31.],
           [ 12., 210., 120.],
           [  4.,  60.,  25.],
           [ 11., 230.,  80.],
           [ 15., 225.,  73.],
           [  2., 110.,  43.]])



### Exercise

Transform given data such that each feature lies in range (0, 1). Use numpy only. 



```python
## Try in this cell

```


```python

```


```python
## Solution
(data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
```




    array([[0.25      , 0.55721393, 0.15555556],
           [0.0625    , 0.29850746, 0.15555556],
           [0.6875    , 0.25373134, 0.33777778],
           [0.6875    , 0.27363184, 0.05333333],
           [0.75      , 0.52238806, 0.14666667],
           [0.1875    , 0.25373134, 0.07555556],
           [0.4375    , 0.25373134, 0.05777778],
           [0.3125    , 0.37313433, 0.06666667],
           [0.875     , 0.74626866, 0.06666667],
           [1.        , 1.        , 1.        ],
           [1.        , 0.34825871, 0.05777778],
           [0.75      , 0.7960199 , 0.4       ],
           [0.8125    , 0.82089552, 0.35555556],
           [0.        , 0.        , 0.11111111],
           [0.3125    , 0.09950249, 0.02666667],
           [0.6875    , 0.7960199 , 0.42222222],
           [0.1875    , 0.04975124, 0.        ],
           [0.625     , 0.89552239, 0.24444444],
           [0.875     , 0.87064677, 0.21333333],
           [0.0625    , 0.29850746, 0.08      ]])




```python

```

### Exercise 
Sklearn provides a function for exactly what you implemented above. Look it up and write code below- this time use sklearn.


```python
## Your solution in this cell

```


```python

```


```python
## Solution
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit_transform(data)
```




    array([[0.25      , 0.55721393, 0.15555556],
           [0.0625    , 0.29850746, 0.15555556],
           [0.6875    , 0.25373134, 0.33777778],
           [0.6875    , 0.27363184, 0.05333333],
           [0.75      , 0.52238806, 0.14666667],
           [0.1875    , 0.25373134, 0.07555556],
           [0.4375    , 0.25373134, 0.05777778],
           [0.3125    , 0.37313433, 0.06666667],
           [0.875     , 0.74626866, 0.06666667],
           [1.        , 1.        , 1.        ],
           [1.        , 0.34825871, 0.05777778],
           [0.75      , 0.7960199 , 0.4       ],
           [0.8125    , 0.82089552, 0.35555556],
           [0.        , 0.        , 0.11111111],
           [0.3125    , 0.09950249, 0.02666667],
           [0.6875    , 0.7960199 , 0.42222222],
           [0.1875    , 0.04975124, 0.        ],
           [0.625     , 0.89552239, 0.24444444],
           [0.875     , 0.87064677, 0.21333333],
           [0.0625    , 0.29850746, 0.08      ]])




```python

```

### [Feature Selection](https://en.wikipedia.org/wiki/Feature_selection)
To predict whether it is going to rain tomorrow has nothing to do with what color of shirt Ali is wearing tomorrow. To your algorithm you should feed features which are relevant. Another commmon example is say you want to predict when Ali arrives at his school. One feature is time when Ali wakes up. Another feature is when the puntual professor arrives at school, however, professor has arrived at 7 A.M. sharp for the past 10 years. There are some other features as well. The point to be noted here is that professor's arrival time is a useless feature since it doesn't change. In your dataset, it's always: 7AM, 7AM... In other words, it's [variance](https://en.wikipedia.org/wiki/Variance) of zero. It'll not help predict Ali's arrival time. How do you remove such features. We can find variance and if it's low, remove it. Sklearn provides [this functionality](https://scikit-learn.org/stable/modules/feature_selection.html) and we'll use it.

Let's construct simple synthetic features.  


```python
X = [[7, 0, 13], [7, 1, 20], [7, 0, 10], [7, 1, 31], [7, 1, 104], [7, 1, 1]]
## The first column is professors arrival time, and we want to remove it.

# import the function 
from sklearn.feature_selection import VarianceThreshold

sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
sel.fit_transform(X)
```




    array([[  0,  13],
           [  1,  20],
           [  0,  10],
           [  1,  31],
           [  1, 104],
           [  1,   1]])



The professor's arrival time has automatically been removed.


```python

```

### Exercise:

Consturct a toy dataset and remove features from it using sklearn. Feel free to explore the documentation and more techniques of feaure selection based on other than variance.


```python

```


### Fewer Features are better.
Also, note that we always want smallest possible set of features. If you can achieve accuracy x with 5 features, then there is not point of using 50 features to achieve the same accuracy x. In 

Let's now consider another example in which we load a dataset 

### Read data into Pandas
Reading data from a csv file into pandas dataframe is as simple as:

[(Docs)](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)


```python
df = pd.read_csv('/Users/hamzaliaqet/Desktop/proposal/racing_data_2_months.csv')
```


```python
raw_features = df.loc[:, df.columns != 'ACTUAL_PROBABILITY']
```


```python
raw_features.head()
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
      <th>date</th>
      <th>runner</th>
      <th>meeting</th>
      <th>race</th>
      <th>barrier</th>
      <th>number</th>
      <th>win_odds_probability</th>
      <th>scratched_at</th>
      <th>overall</th>
      <th>class</th>
      <th>...</th>
      <th>d_previous_price</th>
      <th>d_rating</th>
      <th>d_last_start_rating</th>
      <th>d_predicted</th>
      <th>d_rating_alt</th>
      <th>e_rank</th>
      <th>e_score</th>
      <th>e_percentage</th>
      <th>runner_key</th>
      <th>race_key</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-02-02</td>
      <td>Dignity Bay</td>
      <td>Bunbury</td>
      <td>1</td>
      <td>1.0</td>
      <td>1</td>
      <td>0.133333</td>
      <td>NaN</td>
      <td>8 Starts 0w 2p (0%w 25%p)</td>
      <td>3Y+MSW</td>
      <td>...</td>
      <td>0.066225</td>
      <td>15.0</td>
      <td>20.5</td>
      <td>0.090909</td>
      <td>84.5</td>
      <td>2.0</td>
      <td>17030.0</td>
      <td>0.83</td>
      <td>2020-02-02-Bunbury-1-1</td>
      <td>2020-02-02-Bunbury-1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-02-02</td>
      <td>Tokyo Lad</td>
      <td>Bunbury</td>
      <td>1</td>
      <td>7.0</td>
      <td>2</td>
      <td>0.037037</td>
      <td>NaN</td>
      <td>4 Starts 0w 0p (0%w 0%p)</td>
      <td>3Y+MSW</td>
      <td>...</td>
      <td>0.022026</td>
      <td>-3.0</td>
      <td>25.5</td>
      <td>0.076923</td>
      <td>77.5</td>
      <td>5.0</td>
      <td>6855.0</td>
      <td>0.28</td>
      <td>2020-02-02-Bunbury-1-2</td>
      <td>2020-02-02-Bunbury-1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-02-02</td>
      <td>Agent Jay</td>
      <td>Bunbury</td>
      <td>1</td>
      <td>5.0</td>
      <td>3</td>
      <td>0.166667</td>
      <td>NaN</td>
      <td>2 Starts 0w 1p (0%w 50%p)</td>
      <td>3Y+MSW</td>
      <td>...</td>
      <td>0.140845</td>
      <td>31.0</td>
      <td>30.0</td>
      <td>0.217391</td>
      <td>147.5</td>
      <td>3.0</td>
      <td>13715.0</td>
      <td>0.65</td>
      <td>2020-02-02-Bunbury-1-3</td>
      <td>2020-02-02-Bunbury-1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-02-02</td>
      <td>Tallinn</td>
      <td>Bunbury</td>
      <td>1</td>
      <td>8.0</td>
      <td>4</td>
      <td>0.009901</td>
      <td>NaN</td>
      <td>0 Starts 0w 0p</td>
      <td>3Y+MSW</td>
      <td>...</td>
      <td>NaN</td>
      <td>25.0</td>
      <td>0.0</td>
      <td>0.024390</td>
      <td>52.5</td>
      <td>9.0</td>
      <td>1555.0</td>
      <td>0.00</td>
      <td>2020-02-02-Bunbury-1-4</td>
      <td>2020-02-02-Bunbury-1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-02-02</td>
      <td>Lady Hatras</td>
      <td>Bunbury</td>
      <td>1</td>
      <td>3.0</td>
      <td>5</td>
      <td>0.007937</td>
      <td>NaN</td>
      <td>8 Starts 0w 1p (0%w 13%p)</td>
      <td>3Y+MSW</td>
      <td>...</td>
      <td>0.027397</td>
      <td>21.0</td>
      <td>10.0</td>
      <td>0.076923</td>
      <td>78.0</td>
      <td>7.0</td>
      <td>6000.0</td>
      <td>0.24</td>
      <td>2020-02-02-Bunbury-1-5</td>
      <td>2020-02-02-Bunbury-1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>




```python
labels = df.loc[:,'ACTUAL_PROBABILITY']
```


```python
labels
```




    0        0.071741
    1        0.014286
    2        0.122191
    3        0.003669
    4        0.001779
               ...   
    34969    0.015385
    34970         NaN
    34971    0.066144
    34972    0.024453
    34973         NaN
    Name: ACTUAL_PROBABILITY, Length: 34974, dtype: float64



### Exercise

Play with this dataset. Try scaling features, normalizing. One hot encoding for categorical variables etc.


```python

```
