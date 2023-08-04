import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_excel(r'train_data.xlsx.xlsx')

X_, y = df['Причина'], df['Код причины'] # x- text, y- id

# text to numbers - Bag of Words (BoW)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X_)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = SVC(kernel='linear')
model.fit(X_train, y_train)

predicted_category = model.predict(X_test)
accuracy = accuracy_score(predicted_category, y_test)

df['predictSVC'] = model.predict(X)