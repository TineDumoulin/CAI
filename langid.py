import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

df =pd.read_csv('langid.csv', header=0)
X = df['doc']
y = df['language']

# print(type(X))
# print(type(y))
# print(X[:10])
# print(y[:10])
# print(len(X))
# print(len(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=44, stratify=y)


cv = CountVectorizer()
knn = KNeighborsClassifier()

pipe = Pipeline(steps=[('cv', cv), ('knn', knn)])
parameters = {'cv__ngram_range': [(1,1), (1,2), (2,2)], 'knn__n_neighbors': [3, 5, 7], 'knn__metric': ['manhattan', 'cosine']} # without taking into account CV, the experiment will run 3*3*3=27 times
# wasn't working with euclidian distancefunction, why?

GSCV = GridSearchCV(pipe, parameters, iid=False, cv=2, return_train_score=False, verbose=10)
GSCV.fit(X_train, y_train)
print("Best parameter (CV score=%0.3f):" % GSCV.best_score_)
print(GSCV.best_params_)