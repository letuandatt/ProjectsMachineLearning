import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

data = pd.read_csv("poker-hand.csv", header=None)
data_X = data.iloc[:, :-1]
data_y = data.iloc[:, -1]

kf = KFold(n_splits=10, shuffle=True)

fc_avg_knn = 0
fc_avg_tree = 0
fc_avg_rd = 0
rd_features_importance = 0
cnt = 0

for train_idx, test_idx in kf.split(data_X):
    print(f"Lần lặp thứ {cnt + 1}:")
    data_train_X, data_test_X = data_X.iloc[train_idx, ], data_X.iloc[test_idx, ]
    data_train_y, data_test_y = data_y.iloc[train_idx], data_y.iloc[test_idx]

    # print(f"    Số lượng tập train: {len(data_train_X)}")
    # print(f"    Số lượng tập test: {len(data_test_X)}")

    # --------

    Model_KNN = KNeighborsClassifier(n_neighbors=11)
    Model_KNN.fit(data_train_X, data_train_y)

    data_pred_knn = Model_KNN.predict(data_test_X)

    rate_knn_fc = round(f1_score(data_test_y, data_pred_knn, average='micro') * 100, 2)
    print(f"    Độ chính xác KNN: {rate_knn_fc}")
    fc_avg_knn += rate_knn_fc

    # --------

    Model_Tree = DecisionTreeClassifier(criterion="entropy")
    Model_Tree.fit(data_train_X, data_train_y)

    data_pred_tree = Model_Tree.predict(data_test_X, check_input=True)

    rate_tree_fc = round(f1_score(data_test_y, data_pred_tree, average='micro') * 100, 2)
    print(f"    Độ chính xác DT: {rate_tree_fc}")
    fc_avg_tree += rate_tree_fc

    # --------

    Model_RD = RandomForestClassifier(criterion="entropy")
    Model_RD.fit(data_train_X, data_train_y)

    data_pred_rd = Model_RD.predict(data_test_X)

    rate_rd_fc = round(f1_score(data_test_y, data_pred_rd, average='micro') * 100, 2)
    print(f"    Độ chính xác RD: {rate_rd_fc}\n")
    fc_avg_rd += rate_rd_fc

    rd_features_importance += Model_RD.feature_importances_

    # --------

    cnt += 1

print(f"Độ chính xác trung bình f1 KNN: {fc_avg_knn // cnt}")
print(f"Độ chính xác trung bình f1 DT: {fc_avg_tree // cnt}")
print(f"Độ chính xác trung bình f1 RD: {fc_avg_rd // cnt}")

# Biểu đồ dữ liệu
value_counts_train = data_y.value_counts()
label_name_train = []
number_of_labels_train = []
j = 0

for value, count in value_counts_train.items():
    label_name_train.append(j)
    number_of_labels_train.append(count)
    j += 1

plt.bar(label_name_train, number_of_labels_train)
plt.title('Number of each type of train label')
plt.xlabel('Label type')
plt.ylabel('Label quantity')

for i, value1 in enumerate(number_of_labels_train):
    plt.text(i, value1, str(value1), ha='center', va='bottom')
plt.show()

# Biều đồ độ chính xác
categories = ['KNN', 'DT', 'RD']
values = [fc_avg_knn // cnt, fc_avg_tree // cnt, fc_avg_rd // cnt]

plt.bar(categories, values)
plt.title('Accuracy score of 3 algorithms')

plt.xlabel('Algorithm name')
plt.ylabel('Precision value')

for i, value in enumerate(values):
    plt.text(i, value, str(value), ha='center', va='bottom')

plt.show()

# Biểu đồ thuộc tính quan trọng
feature_names = ['Chất lá 1', 'Số lá 1', 'Chất lá 2', 'Số lá 2', 'Chất lá 3', 'Số lá 3', 'Chất lá 4', 'Số lá 4', 'Chất lá 5', 'Số lá 5']
plt.figure(figsize=(10, 6))
plt.barh(feature_names, rd_features_importance / cnt)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance")
plt.show()

# Cây => Luật
# tree_rules = export_text(Model_Tree, feature_names=feature_names)

def extract_rules(tree, feature_names):
    rules = []
    stack = [(0, -1)]  # Dùng một ngăn xếp để theo dõi node và depth
    while stack:
        node, depth = stack.pop()
        if node < 0:
            continue
        if depth > 0:
            if tree.tree_.children_left[node] == tree.tree_.children_right[node]:
                value = np.argmax(tree.tree_.value[node])
                rule = "THEN Class {} (leaf node)".format(value)
            else:
                rule = "IF {} <= {}".format(feature_names[tree.tree_.feature[node]], tree.tree_.threshold[node])
            rules.append("{}{}".format("  " * depth, rule))
        stack.append((tree.tree_.children_left[node], depth + 1))
        stack.append((tree.tree_.children_right[node], depth + 1))
    return rules

tree_rules = extract_rules(Model_Tree, feature_names)
print(tree_rules)