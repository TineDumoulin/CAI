import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('vacatures_train_dutch.csv', header=0)
description = df['description'].values
label = df['type'].values

description_train, description_test, y_train, y_test = train_test_split(description, label, test_size=0.25, random_state=1000)

vectorizer = CountVectorizer() 
vectorizer.fit(description_train)

X_train = vectorizer.transform(description_train) 
X_test  = vectorizer.transform(description_test)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)

print("Accuracy:", score)
# resulted in 'Accuracy: 0.7508480325644504'