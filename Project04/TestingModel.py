import LoadData as ld
import joblib as jl

from sklearn.metrics import accuracy_score

def testing():
    data_train_X, data_test_X, data_train_y, data_test_y = ld.hold_out()

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

    return knn_acc, bayes_acc, tree_acc, gd_acc