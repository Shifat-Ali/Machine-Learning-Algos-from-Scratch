import numpy as np
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt

def gini_impurity(y):
    freq = np.bincount(y)
    ps = freq/len(y)
    return 1 - sum(ps*ps)

def weighted_ginix(X_column, y, threshold):
    left_idxs = np.argwhere(X_column <= threshold).flatten()
    right_idxs = np.argwhere(X_column > threshold).flatten()

    n = len(y)
    n_left, n_right = len(left_idxs), len(right_idxs)
    gi_left, gi_right = gini_impurity(y[left_idxs]), gini_impurity(y[right_idxs])

    return (n_left/n)*gi_left + (n_right/n)*gi_right

data = datasets.load_wine()
X,y = data.data, data.target

feature_names = data.feature_names
impurity_data = []
min_impurity = 2
best_feature, best_threshold = None, None

for i,f in enumerate(feature_names):
    for threshold in np.unique(X[:,i]):
        gidx = weighted_ginix(X[:,i], y, threshold)
        impurity_data.append([f, threshold, gidx])
        if gidx < min_impurity:
            min_impurity = gidx
            best_feature = f
            best_threshold = threshold

        # print(f'Gini Impurity for {f} with threshold {threshold} = {gidx}')

impurity_df = pd.DataFrame(impurity_data, columns=['feature', 'threshold', 'impurity'])
# print(impurity_df)
print('Best Feature: ', best_feature)
print('Best Threshold: ', best_threshold)
print('Minimum Gini Index: ', min_impurity)

grouped_imp_df = impurity_df.groupby(by='feature').min()
print(grouped_imp_df)
grouped_imp_df.impurity *= 100
grouped_imp_df.plot(kind='bar')