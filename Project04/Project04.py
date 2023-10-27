import pandas as pd

data = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/mushrooms.csv")
data_X = data.drop("class", axis=1)
data_Y = data.iloc[:, 0]

