import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv")
data_X = data.drop("logS", axis=1)
data_y = data.logS

data_train_X, data_test_X, data_train_y, data_test_y = train_test_split(data_X, data_y, test_size=0.2, random_state=100)

# LINEAR REGRESSION
lr = LinearRegression()
lr.fit(data_train_X, data_train_y)

data_pred_lr_train = lr.predict(data_train_X)
data_pred_lr_test = lr.predict(data_test_X)

lr_train_mse = mean_squared_error(data_train_y, data_pred_lr_train)
lr_train_r2 = r2_score(data_train_y, data_pred_lr_train)

lr_test_mse = mean_squared_error(data_test_y, data_pred_lr_test)
lr_test_r2 = r2_score(data_test_y, data_pred_lr_test)

lr_results = pd.DataFrame(['Linear Regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method', 'Traning MSE', 'Training R2', 'Test MSE', 'Test R2']

# KNN
knn = KNeighborsRegressor(n_neighbors=9)
knn.fit(data_train_X, data_train_y)

data_pred_knn_train = knn.predict(data_train_X)
data_pred_knn_test = knn.predict(data_test_X)

knn_train_mse = mean_squared_error(data_train_y, data_pred_knn_train)
knn_train_r2 = r2_score(data_train_y, data_pred_knn_train)

knn_test_mse = mean_squared_error(data_test_y, data_pred_knn_test)
knn_test_r2 = r2_score(data_test_y, data_pred_knn_test)

knn_results = pd.DataFrame(['KNeighborsRegressor', knn_train_mse, knn_train_r2, knn_test_mse, knn_test_r2]).transpose()
knn_results.columns = ['Method', 'Traning MSE', 'Training R2', 'Test MSE', 'Test R2']

# RANDOM FOREST
rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(data_train_X, data_train_y)

data_pred_rf_train = rf.predict(data_train_X)
data_pred_rf_test = rf.predict(data_test_X)

rf_train_mse = mean_squared_error(data_train_y, data_pred_rf_train)
rf_train_r2 = r2_score(data_train_y, data_pred_rf_train)

rf_test_mse = mean_squared_error(data_test_y, data_pred_rf_test)
rf_test_r2 = r2_score(data_test_y, data_pred_rf_test)

rf_results = pd.DataFrame(['Random Forest', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
rf_results.columns = ['Method', 'Traning MSE', 'Training R2', 'Test MSE', 'Test R2']

# MODEL COMPARISON
df_models = pd.concat([lr_results, knn_results, rf_results], axis=0).reset_index(drop=True)
print(df_models)

# VISUALIZATION

plt.figure(figsize=(5, 5))
plt.scatter(data_train_y, data_pred_lr_train, alpha=0.3)
plt.title("Prediction by LinearRegression")
z = np.polyfit(data_train_y, data_pred_lr_train, 1)
p = np.poly1d(z)

plt.plot(data_train_y, p(data_train_y), '#F8766D')
plt.xlabel('Experimental LogS')
plt.ylabel('Predict LogS')
plt.show()


plt.figure(figsize=(5, 5))
plt.scatter(data_train_y, data_pred_knn_train, alpha=0.3)
plt.title("Prediction by K-Nearest Neighbors")
z = np.polyfit(data_train_y, data_pred_knn_train, 1)
p = np.poly1d(z)

plt.plot(data_train_y, p(data_train_y), '#F8766D')
plt.xlabel('Experimental LogS')
plt.ylabel('Predict LogS')
plt.show()