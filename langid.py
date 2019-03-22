import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# part 1
df =pd.read_csv('langid.csv', header=0)
X = df['doc']
y = df['language']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=44, stratify=y)
# wasn't working without 'stratify=y', why?

# part 2
cv = CountVectorizer()
knn = KNeighborsClassifier()
pipe = Pipeline(steps=[('cv', cv), ('knn', knn)])

# part 3
parameters = {'cv__ngram_range': [(1,1), (1,2), (2,2)], 'knn__n_neighbors': [3, 5, 7], 'knn__metric': ['manhattan', 'cosine']} 
# without taking into account crossval, the experiment will run 3*3*2=18 times
# wasn't working with euclidian distance function, why?

# part 4
GSCV = GridSearchCV(pipe, parameters, iid=False, cv=10, return_train_score=False, verbose=10)
GSCV.fit(X_train, y_train)
print("Best parameter (CV score=%0.3f):" % GSCV.best_score_)
print(GSCV.best_params_)

# Best parameter (CV score=0.744):
# {'cv__ngram_range': (1, 1), 'knn__metric': 'cosine', 'knn__n_neighbors': 3}

""" # part 5
cv = CountVectorizer(ngram_range=(1,1))
knn = KNeighborsClassifier(n_neighbors=3, metric='cosine')
pipe = Pipeline(steps=[('cv', cv), ('knn', knn)])

cv_scores_train = cross_val_score(pipe, X, y, cv=10, verbose=5)
print('cv_scores mean of train data:{}'.format(np.mean(cv_scores_train)))

# part 6
pipe.fit(X, y)
y_pred = pipe.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) """