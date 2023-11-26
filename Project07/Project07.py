import pandas as pd
import joblib as jl
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

# --------

data = pd.read_csv("car.data", header=None)
data.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
print(data)

# --------

X = data.drop("class", axis=1)
y = data.iloc[:, -1]

label_encoder = LabelEncoder()

for column in X.columns:
    X[column] = label_encoder.fit_transform(X[column])

y = pd.Series(label_encoder.fit_transform(y))
y.name = 'class'

print(X.join(y))

# --------

labels = ['unacc', 'acc', 'good', 'v-good']
sizes = [1210, 384, 69, 65]
plt.figure(figsize=(7, 8))
plt.title("Biểu đồ % các nhãn")
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.legend(labels, loc="best")
plt.axis('equal')
plt.show()

# --------

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train_data = X_train.join(y_train)
# train_data.hist(figsize=(20, 15))
# plt.show()
#
# plt.figure(figsize=(20, 10))
# sb.heatmap(train_data.corr(), annot=True, cmap='YlGnBu')
# plt.show()

# --------

knn_avg = 0
rnc_avg = 0
bayes_avg = 0
tree_avg = 0
rd_avg = 0
svm_avg = 0
mlp_avg = 0
xgb_avg = 0
cnt = 0

kf = KFold(n_splits=50, shuffle=True)

for train_idx, test_idx in kf.split(X):
    print(f"Chia lần {cnt + 1}:")

    X_train, X_test = X.iloc[train_idx, ], X.iloc[test_idx, ]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_train, y_train)
    pred_knn = knn.predict(X_test)
    f1_knn = f1_score(y_test, pred_knn, average='micro') * 100
    knn_avg += f1_knn
    print(f"    F1 score of KNeighborsClassifier: {f1_knn}")

    rnc = RadiusNeighborsClassifier(radius=1.0)
    rnc.fit(X_train, y_train)
    pred_rnc = rnc.predict(X_test)
    f1_rnc = f1_score(y_test, pred_rnc, average='micro') * 100
    rnc_avg += f1_rnc
    print(f"    F1 score of RadiusNeighborsClassifier: {f1_rnc}")

    bayes = CategoricalNB()
    bayes.fit(X_train, y_train)
    pred_bayes = bayes.predict(X_test)
    f1_bayes = f1_score(y_test, pred_bayes, average='micro') * 100
    bayes_avg += f1_bayes
    print(f"    F1 score of CategoricalNB: {f1_bayes}")

    tree = DecisionTreeClassifier(criterion="entropy")
    tree.fit(X_train, y_train)
    pred_tree = tree.predict(X_test)
    f1_tree = f1_score(y_test, pred_tree, average='micro') * 100
    tree_avg += f1_tree
    print(f"    F1 score of DecisionTreeClassifier: {f1_tree}")

    rd = RandomForestClassifier(criterion="entropy")
    rd.fit(X_train, y_train)
    pred_rd = rd.predict(X_test)
    f1_rd = f1_score(y_test, pred_rd, average='micro') * 100
    rd_avg += f1_rd
    print(f"    F1 score of RandomForestClassifier: {f1_rd}")

    svm_model = svm.SVC(kernel="rbf", gamma="auto")
    svm_model.fit(X_train, y_train)
    pred_svm = svm_model.predict(X_test)
    f1_svm = f1_score(y_test, pred_svm, average='micro') * 100
    svm_avg += f1_svm
    print(f"    F1 score of SupportVectorMachine: {f1_svm}")

    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, activation='relu')
    mlp.fit(X_train, y_train)
    pred_mlp = mlp.predict(X_test)
    f1_mlp = f1_score(y_test, pred_mlp, average='micro') * 100
    mlp_avg += f1_mlp
    print(f"    F1 score of Multilayer Perceptron: {f1_mlp}")

    xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    xgb_model.fit(X_train, y_train)
    pred_xgb_model = xgb_model.predict(X_test)
    f1_xgb_model = f1_score(y_test, pred_xgb_model, average='micro') * 100
    xgb_avg += f1_xgb_model
    print(f"    F1 score of XGBoostClassifier: {f1_xgb_model}")

    print("---")
    cnt += 1

print(f"Avg F1 score of KNeighborsClassifier: {knn_avg // cnt}")
print(f"Avg F1 score of RadiusNeighborsClassifier: {rnc_avg // cnt}")
print(f"Avg F1 score of Bayes: {bayes_avg // cnt}")
print(f"Avg F1 score of DecistionTree: {tree_avg // cnt}")
print(f"Avg F1 score of RandomForestClassifier: {rd_avg // cnt}")
print(f"Avg F1 score of SupportVectorMachine: {svm_avg // cnt}")
print(f"Avg F1 score of Multilayer Perceptron: {mlp_avg // cnt}")
print(f"Avg F1 score of XGBoostClassifier: {xgb_avg // cnt}")

# --------

classes = ["KNN", "RNC", "Bayes", "DT", "RD", "SVM", "MLP", "XGB"]
counts = [knn_avg // cnt, rnc_avg // cnt, bayes_avg // cnt, tree_avg // cnt, rd_avg // cnt, svm_avg // cnt, mlp_avg // cnt, xgb_avg // cnt]

data = pd.DataFrame({'Class': classes, 'Count': counts})

ax = sb.barplot(x='Mô hình', y='Độ chính xác', data=data, palette='viridis', legend=False)

for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.title("Accuracy score of 8 Models")
plt.show()