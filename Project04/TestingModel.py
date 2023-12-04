import LoadData as ld
import joblib as jl

from sklearn.metrics import accuracy_score

_, data_test_X, _, data_test_y = ld.hold_out()
def testing():
    knn_model = jl.load("knn_model.pkl")
    knn_pred = knn_model.predict(data_test_X)
    knn_acc = accuracy_score(knn_pred, data_test_y) * 100

    bayes_model = jl.load("bayes_model.pkl")
    bayes_pred = bayes_model.predict(data_test_X)
    bayes_acc = accuracy_score(bayes_pred, data_test_y) * 100

    tree_model = jl.load("tree_model.pkl")
    tree_pred = tree_model.predict(data_test_X)
    tree_acc = accuracy_score(tree_pred, data_test_y) * 100

    gd_model = jl.load("gradientboosting_model.pkl")
    gd_pred = gd_model.predict(data_test_X)
    gd_acc = accuracy_score(gd_pred, data_test_y) * 100

    mlp_model = jl.load("MLPClassifier.pkl")
    mlp_pred = mlp_model.predict(data_test_X)
    mlp_acc = accuracy_score(mlp_pred, data_test_y) * 100

    return knn_acc, bayes_acc, tree_acc, gd_acc, mlp_acc