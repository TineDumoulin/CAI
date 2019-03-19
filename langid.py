import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('langid.csv', header=0)
X = df['doc']
y = df['language']

# print(type(X))
# print(type(y))
# print(X[:10])
# print(y[:10])
# print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=44)
# print(type(X_train), type(y_train))

cv = CountVectorizer(analyzer='ngrams')
knn = KNeighborsClassifier()

pipe = Pipeline([("cv", cv), ("knn", knn)])
parameters = {'cv__ngram_range': [(1,1), (2,2), (3,3)], 'knn__n_neighbors': [3, 5, 7], 'knn__metric': ['manhattan', 'euclidian', 'chebyshev']} # without taking into account CV, the experiment will run 3*3*2=18 times

GSCV = GridSearchCV(pipe, parameters, iid=False, cv=10, return_train_score=False)
GSCV.fit(X_train, y_train)
print("Best parameter (CV score=%0.3f):" % GSCV.best_score_)
print(GSCV.best_params_)

# I keep getting this error:
# ValueError: ngrams is not a valid tokenization scheme/analyzer
# and I've tried looking up how to fix it, but it's not working.