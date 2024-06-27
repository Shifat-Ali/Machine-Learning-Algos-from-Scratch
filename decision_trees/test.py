from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from DecisionTree import DecisionTree
import matplotlib.pyplot as plt

data = datasets.load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTree(max_depth=10)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred)/len(y_test)

acc = accuracy(y_test, predictions)
print(acc)

def plot():
    avg_acc = []
    for i in range(100):
        clf = DecisionTree(max_depth=10)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)



        acc = accuracy(y_test, predictions)
        avg_acc.append(acc)

    avg_acc = np.array(avg_acc)
    mean = avg_acc.mean()
    std = avg_acc.std()
    print('Shape of dataset', X.shape)
    print('Mean of acc:', mean)
    print('Std dev of acc:', std)
    print('Variance of acc:', avg_acc.var())
    plt.plot([i for i in range(1, len(avg_acc)+1)], avg_acc)
    plt.axhline(mean, label = 'Mean')
    plt.axhline(mean+3*std, label='75')
    plt.axhline(mean-3*std, label='25')
    plt.legend()
    plt.show()