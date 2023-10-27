import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold

data = load_breast_cancer()
data_X = data.data
data_y = data.target

kf = KFold(n_splits=30, shuffle=True)

