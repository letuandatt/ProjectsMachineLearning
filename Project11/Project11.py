import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# SEPERATING X & Y
data = pd.read_csv("housing.csv")
data_X = data.drop('median_house_value', axis=1)
data_y = data.median_house_value

# DATA INFORMATION
# print(data.info())
# data.dropna(inplace=True)
# print(data.info())

# DATA EXPLORATION
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2)

train_data = X_train.join(y_train)
# print(train_data)
#
# print(train_data.hist(figsize=(15, 8)))
# plt.show()
#
# plt.figure(figsize=(15, 8))
# sb.heatmap(train_data.corr(), annot=True, cmap="YlGnBu")

# # Not done

# DATA PREPROCESSING
train_data['total_rooms'] = np.log(train_data['total_rooms'] + 1)
train_data['total_bedrooms'] = np.log(train_data['total_bedrooms'] + 1)
train_data['population'] = np.log(train_data['population'] + 1)
train_data['households'] = np.log(train_data['households'] + 1)

# train_data.hist(figsize=(15, 8))
# plt.show()

train_data = train_data.join(pd.get_dummies(train_data.ocean_proximity).drop(train_data.ocean_proximity, axis=1))
print(train_data)