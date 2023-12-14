import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# SPLITTING THE DATA TO X AND Y FROM FIRST DATASET
data = pd.read_csv("wdbc.data", header=None)
data_X = data.iloc[:, 2:]
data_y = data.iloc[:, 1]

# SPLITTING DATA TO X AND Y FOR TRAINING AND TESTING
kf = KFold(n_splits=10, shuffle=True)

knn_avg_acc = 0
tree_avg_acc = 0
rd_avg_acc = 0
gb_avg_acc = 0
cnt = 0

# TRAINING AND TESTING
for train_idx, test_idx in kf.split(data_X):
    print(f"Lần lặp thứ {cnt + 1}:")

    data_train_X, data_test_X = data_X.iloc[train_idx, ], data_X.iloc[test_idx, ]
    data_train_y, data_test_y = data_y.iloc[train_idx], data_y.iloc[test_idx]

    print(f"    Size training: {len(data_train_X)}")
    print(f"    Size testing: {len(data_test_X)}")

    # --------

    knn = KNeighborsClassifier(n_neighbors=11)
    knn.fit(data_train_X, data_train_y)

    knn_pred = knn.predict(data_test_X)

    knn_acc = round(accuracy_score(data_test_y, knn_pred) * 100, 2)
    print(f"    Độ chính xác acc KNN: {knn_acc}")
    knn_avg_acc += knn_acc

    # -------

    tree = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    tree.fit(data_train_X, data_train_y)

    tree_pred = tree.predict(data_test_X)

    tree_acc = round(accuracy_score(data_test_y, tree_pred) * 100, 2)
    print(f"    Độ chính xác acc DT: {tree_acc}")
    tree_avg_acc += tree_acc

    # -------

    rd = RandomForestClassifier(criterion="entropy", n_estimators=50, min_samples_leaf=3)
    rd.fit(data_train_X, data_train_y)

    rd_pred = rd.predict(data_test_X)

    rd_acc = round(accuracy_score(data_test_y, tree_pred) * 100, 2)
    print(f"    Độ chính xác acc RD: {rd_acc}")
    rd_avg_acc += rd_acc

    # -------

    gb = GradientBoostingClassifier()
    gb.fit(data_train_X, data_train_y)

    gb_pred = gb.predict(data_test_X)

    gb_acc = round(accuracy_score(data_test_y, gb_pred) * 100, 2)
    print(f"    Độ chính xác acc GB: {gb_acc}")
    gb_avg_acc += gb_acc

    # -------

    cnt += 1

print(f"\nĐộ chính xác trung bình KNN acc: {knn_avg_acc // cnt}")
print(f"Độ chính xác trung bình DT acc: {tree_avg_acc // cnt}")
print(f"Độ chính xác trung bình RD acc: {rd_avg_acc // cnt}")
print(f"Độ chính xác trung bình GB acc: {gb_avg_acc // cnt}")

# VISUALIZE ACCURACY SCORE OF 4 ALGORITHMS
categories = ['KNN', 'DT', 'RD', 'GB']
values = [knn_avg_acc // cnt, tree_avg_acc // cnt, rd_avg_acc // cnt, gb_avg_acc // cnt]

plt.bar(categories, values)
plt.title('Accuracy score of 4 algorithms')

plt.xlabel('Algorithm name')
plt.ylabel('Precision value')

for i, value in enumerate(values):
    plt.text(i, value, str(value), ha='center', va='bottom')

plt.show()