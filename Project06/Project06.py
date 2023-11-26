import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# DATA COLLECTION AND PRE-PROCESSING
data = pd.read_csv("mail_data.csv")

data_ = data.where((pd.notnull(data)), '')

# print(data.head())
# print(data_.shape)

# LABEL ENCODING
data_.loc[data_['Category'] == 'spam', 'Category', ] = 0
data_.loc[data_['Category'] == 'ham', 'Category', ] = 1

# print(data_)

X = data_['Message']
y = data_['Category']

# SPLITTING THE DATA INTO TRAINING AND TESTING
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

print(X.shape)
print(X_train.shape)
print(X_test.shape)

# FEATURE EXTRACTION
feature_extraction = TfidfVectorizer(min_df = 1, stop_words = 'english', lowercase= True)
X_train = feature_extraction.fit_transform(X_train)
X_test = feature_extraction.transform(X_test)

y_train = y_train.astype('int')
y_test = y_test.astype('int')

# print(X_train)

# TRAINING MODEL
model = LogisticRegression()
model.fit(X_train, y_train)

# EVALUATING TRAINED MODEL
prediction_on_training = model.predict(X_train)
accuracy_training = accuracy_score(y_train, prediction_on_training) * 100
print(accuracy_training)

prediction_on_testing = model.predict(X_test)
accuracy_testing = accuracy_score(y_test, prediction_on_testing) * 100
print(accuracy_testing)

# NEW DATA COMES
input_mail = ["Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030"]

input_data_features = feature_extraction.transform(input_mail)

prediction = model.predict(input_data_features)

if(prediction[0] == 1):
    print("Ham mail")
else:
    print("Spam mail")