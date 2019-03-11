from sklearn.feature_extraction import DictVectorizer
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.model_selection import cross_val_score

df = pd.read_csv('diminutive.train.csv', header=None)
y = df.iloc[:,-1]

def labeler(pdDF):
    labels = {}
    c = 0  
    for line in (pdDF.iloc[:,-1]):
        if line not in labels:
            labels[line[-1]] = c
            c += 1
    return labels

# y = labeler(df)
lbls = {'T': 0, 'E': 1, 'J': 2, 'P': 3, 'K': 4}

X = df.iloc[:,:-1]
X_train = X.to_dict('records')

dv = DictVectorizer(sparse=False)
knn = KNeighborsClassifier(n_neighbors=3, metric='manhattan')
p = Pipeline([("dv", dv), ("cls", knn)])

cv_scores_train = cross_val_score(p, X_train, y, cv=10)
print(cv_scores_train)
print('cv_scores mean of train data:{}'.format(np.mean(cv_scores_train)))

X_test_todict = pd.read_csv('diminutive.test.csv', header=None).iloc[:,:-1].to_dict('records')
y_test = pd.read_csv('diminutive.test.csv', header=None).iloc[:,-1]

cv_scores_test = cross_val_score(p, X_test_todict, y_test, cv=10)
print(cv_scores_test)
print('cv_scores mean of test data:{}'.format(np.mean(cv_scores_test)))