import pandas as pd

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data_train = pd.read_csv("poker-hand-training-true.data", header=None)
X_train = data_train.iloc[:, : -1]
y_train = data_train.iloc[:, -1]

ros = RandomOverSampler()
X_train, y_train = ros.fit_resample(X_train, y_train)

data_test = pd.read_csv("poker-hand-testing.data", header=None)
X_test = data_test.iloc[:, : -1]
y_test = data_test.iloc[:, -1]

knn = KNeighborsClassifier(n_neighbors=11, n_jobs=4)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
print(acc)