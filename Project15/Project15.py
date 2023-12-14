import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import warnings

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/CarPrice.csv")

data = data[["symboling", "wheelbase", "carlength",
             "carwidth", "carheight", "curbweight",
             "enginesize", "boreratio", "stroke",
             "compressionratio", "horsepower", "peakrpm",
             "citympg", "highwaympg", "price"]]

data.head()
data.isnull().sum()
data.info()
data.describe()

sb.set_style("whitegrid")
plt.figure(figsize=(15, 10))
sb.distplot(data.price)
plt.show()

plt.figure(figsize=(20, 15))
sb.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.show()

X = np.array(data.drop("price", axis=1))
y = np.array(data["price"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(model.score(X_test, y_pred) * 100)