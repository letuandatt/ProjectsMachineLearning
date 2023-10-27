import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import *
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, f1_score

# SPLITTING DATA TO X & Y
data_train = pd.read_csv("poker-hand-training-true.data", header=None)
data_train_X = data_train.iloc[:, :-1]
data_train_y = data_train.iloc[:, -1]

data_test = pd.read_csv("poker-hand-testing.data", header=None)
data_test_X = data_test.iloc[:, :-1]
data_test_y = data_test.iloc[:, -1]

# READING DATA DESCRIPTION
print(f"Number of samples in the training set: {len(data_train)}")
print(f"Number of samples in the test set: {len(data_test)}")
print(f"Labels in the dataset: {np.unique(data_test_y)}")
print(f"Number of each label in the training set:\n{data_train_y.value_counts()}")
print(f"Number of each label in the test set:\n{data_test_y.value_counts()}")
# # Lấy tên thuộc tính

# KNN
start_knn = time.time()

Model_KNN = KNeighborsClassifier(n_neighbors=11)
Model_KNN.fit(data_train_X, data_train_y)

data_pred_train_KNN = Model_KNN.predict(data_train_X)
data_pred_test_KNN = Model_KNN.predict(data_test_X)

rate_train_KNN = accuracy_score(data_train_y, data_pred_train_KNN) * 100
rate_test_KNN = accuracy_score(data_test_y, data_pred_test_KNN) * 100

rate_train_KNN_fc = f1_score(data_train_y, data_pred_train_KNN, average=None)
rate_test_KNN_fc = f1_score(data_train_y, data_pred_train_KNN, average=None)

knn_results = pd.DataFrame(['KNeighborClassifier', round(rate_train_KNN, 2), round(rate_test_KNN, 2)]).transpose()
knn_results.columns = ['Method', 'ACS Train', 'ACS Test']

end_knn = time.time()

# DECISION TREE
start_dt = time.time()

Model_Tree = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3)
Model_Tree.fit(data_train_X, data_train_y)

data_pred_train_tree = Model_Tree.predict(data_train_X)
data_pred_test_tree = Model_Tree.predict(data_test_X)

rate_train_tree = accuracy_score(data_train_y, data_pred_train_tree) * 100
rate_test_tree = accuracy_score(data_test_y, data_pred_test_tree) * 100

rate_train_tree_fc = f1_score(data_train_y, data_pred_train_tree, average=None)
rate_test_tree_fc = f1_score(data_test_y, data_pred_test_tree, average=None)

tree_results = pd.DataFrame(['Decision Tree', round(rate_train_tree, 2), round(rate_test_tree, 2)]).transpose()
tree_results.columns = ['Method', 'ACS Train', 'ACS Test']

end_dt = time.time()

# RANDOM FOREST
start_rd = time.time()

Model_RD = RandomForestClassifier(criterion="entropy", n_estimators=50, random_state=100)
Model_RD.fit(data_train_X, data_train_y)

data_pred_train_rd = Model_RD.predict(data_train_X)
data_pred_test_rd = Model_RD.predict(data_test_X)

rate_train_rd = accuracy_score(data_train_y, data_pred_train_rd) * 100
rate_test_rd = accuracy_score(data_test_y, data_pred_test_rd) * 100

rate_train_rd_fc = f1_score(data_train_y, data_pred_train_rd, average=None)
rate_test_rd_fc = f1_score(data_test_y, data_pred_test_rd, average=None)

rd_results = pd.DataFrame(['Random Forest', round(rate_train_rd, 2), round(rate_test_rd, 2)]).transpose()
rd_results.columns = ['Method', 'ACS Train', 'ACS Test']

end_rd = time.time()

# MODEL COMPARISON (So sánh chỉ số đánh giá các mô hình)
poker_results = pd.concat([knn_results, tree_results, rd_results], axis=0).reset_index(drop=True)
print(poker_results)

# VISUALIZING (Vẽ biểu đồ)

# 1. Biểu đồ chỉ số đánh giá 3 giải thuật
categories = ['KNN','DT', 'RD']
values = [round(rate_test_KNN, 2), round(rate_test_tree, 2), round(rate_test_rd, 2)]

plt.bar(categories, values)

plt.title('Accuracy score of 3 algorithms')

plt.xlabel('algorithm name')
plt.ylabel('precision value')

for i, value in enumerate(values):
    plt.text(i, value, str(value), ha='center', va='bottom')

plt.show()

# 2. Biểu đồ tương quan dữ liệu


# TIME TRAINING & TEST MODEL OF 3 ALGORITHMS
print(f"\nTime for KNN: {end_knn - start_knn}")
print(f"Time for DecisionTree: {end_dt - start_dt}")
print(f"Time for Random Forest: {end_rd - start_rd}")
