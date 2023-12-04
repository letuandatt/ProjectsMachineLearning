import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_data():
    data = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/mushrooms.csv")
    data_X = data.drop("class", axis=1)
    data_y = data.iloc[:, 0]

    label_encoder = LabelEncoder()

    for column in data_X.columns:
        data_X[column] = label_encoder.fit_transform(data_X[column])

    data_y = label_encoder.fit_transform(data_y)
    data_y = pd.Series(data_y)  # Dùng pd.Series(...).value_counts() để in size đã mã hóa

    data_X = pd.concat([data_X.iloc[:, i] for i in range(22)], axis=1)
    data_y.name = 'class'

    data = data_X.join(data_y)
    return data, data_X, data_y

def hold_out():
    _, data_X, data_y = load_data()
    data_train_X, data_test_X, data_train_y, data_test_y = train_test_split(data_X, data_y, test_size=1/3.0,
                                                                            random_state=42)
    return data_train_X, data_test_X, data_train_y, data_test_y