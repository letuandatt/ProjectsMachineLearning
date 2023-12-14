import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# GETTING DATA AND DATA'S INFORMATION
data = pd.read_csv("healthcare.csv")

# data.info()

# PREPROCESSING
le = LabelEncoder()

data['Age'] = (data['Age'] - data['Age'].min()) / (data['Age'].max() - data['Age'].min())
data['Gender'] = le.fit_transform(data['Gender'])
data['Blood_Type'] = le.fit_transform(data['Blood_Type'])
data['Insurance_Provider'] = le.fit_transform(data['Insurance_Provider'])
data['Admission_Type'] = le.fit_transform(data['Admission_Type'])
data['Medication'] = le.fit_transform(data['Medication'])
data['Test_Results'] = le.fit_transform(data['Test_Results'])


data = data[['Age', 'Gender', 'Blood_Type', 'Insurance_Provider', 'Admission_Type', 'Medication', 'Test_Results']]
# print(data.head().iloc[:, 5:])

# TRAIN_TEST_SPLIT
X = data.iloc[:, : -1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3.0, random_state=42)

# DATA EXPLORING
train_data = X_train.join(y_train)

# correlation_matrix = train_data.corr()
# plt.figure(figsize=(10, 8))
# sb.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
# plt.title("Correlation Heatmap of Heathcare")
# plt.show()
#
# plt.figure(figsize=(10, 8))
# sb.boxplot(data=train_data, orient='h')
# plt.title("Box Plots of Heathcare")
# plt.show()

# MODEL
# knn = CategoricalNB()
# knn.fit(X_train, y_train)
#
# y_pred = knn.predict(X_test)
# print(f"Accuracy of KNN Model: {accuracy_score(y_test, y_pred) * 100}")
# print(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred, labels=[np.unique(y_test)])}")

