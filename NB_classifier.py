# 1)
from sklearn.datasets import fetch_20newsgroups

df = fetch_20newsgroups(subset='train', categories=('alt.atheism', 'sci.space'))
X = df.data
y = df.target

# 2) + 3)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(analyzer='char',ngram_range=(3,3),max_features=1000)
# waarom gebruiken we een CountVectorizer? zet data om in lijsten van getallen
X_train = cv.fit_transform(X[:860])
y_train = y[:860]
X_test = cv.fit_transform(X[860:])
y_test = y[860:]


# 4)
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

nb = MultinomialNB()
nb.fit(X_train,y_train)
y_test_pred = nb.predict(X_test)
print(classification_report(y_test,y_test_pred))

# 5)
from sklearn.datasets import fetch_20newsgroups
newsgroups = cv.transform(fetch_20newsgroups(subset='test', categories=('alt.atheism', 'sci.space')))
prediction = nb.predict(newsgroups)
print(prediction)

import numpy as np
np.exp(nb.class_log_prior) # de prior weergeven
np.exp(nb.feature_log_prob) # de weights (probabiliteiten) weergeven