import pandas as pd
from flask import Flask, render_template, request

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Đọc dữ liệu từ file CSV
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/SMS-Spam-Detection/master/spam.csv", encoding="latin-1")
data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

# Mapping nhãn
data['label'] = data['class'].map({'ham': 0, 'spam': 1})
X = data['message']
y = data['label']

# Tạo và fit CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X)

# Chia dữ liệu thành tập train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Tạo và fit model
clf = MultinomialNB()
clf.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_pred = clf.predict(vect)

        return render_template('result.html', prediction=my_pred[0])

if __name__ == '__main__':
    app.run(debug=True)