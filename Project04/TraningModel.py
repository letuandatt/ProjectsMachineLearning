import LoadData as ld
import joblib as jl

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier

data_train_X, _, data_train_y, _ = ld.hold_out()

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(data_train_X, data_train_y)
jl.dump(knn_model, "knn_model.pkl")

bayes_model = GaussianNB()
bayes_model.fit(data_train_X, data_train_y)
jl.dump(bayes_model, "bayes_model.pkl")

bayes1_model = CategoricalNB()
bayes1_model.fit(data_train_X, data_train_y)
jl.dump(bayes1_model, "bayes_model.pkl")

tree_model = DecisionTreeClassifier(criterion="entropy")
tree_model.fit(data_train_X, data_train_y)
jl.dump(tree_model, "tree_model.pkl")

gd_model = GradientBoostingClassifier()
gd_model.fit(data_train_X, data_train_y)
jl.dump(gd_model, "GradientBoosting_model.pkl")

mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, activation='relu')
mlp_model.fit(data_train_X, data_train_y)
jl.dump(mlp_model, "MLPClassifier.pkl")