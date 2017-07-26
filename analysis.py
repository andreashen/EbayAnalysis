#!/usr/bin/env python

import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# read data from *.csv
test_set = pd.read_csv('data/TestSet.csv')
train_set = pd.read_csv('data/TrainingSet.csv')
test_subset = pd.read_csv('data/TestSubset.csv')
train_subset = pd.read_csv('data/TrainingSubset.csv')

# check information of 'train_set'
train_set.info()


print(train_set[:3])
train = train_set.drop(['EbayID', 'QuantitySold', 'SellerName'], axis=1)
train_target = train_set['QuantitySold']

# get total num of features
[_, n_features] = train.shape

# isSold -> 1:success | 0:fail
df = DataFrame(np.hstack(train, train_target[:, None]), \
               columns=range(n_features)+['isSold'])
_ = sns.pairplot(df[:50], vars=[2, 3, 4, 10, 13], hue='isSold', size=1.5)
plt.figure(figsize=(10, 10))
# caculate correlation matrix
corr = df.corr()
