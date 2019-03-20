from sklearn.feature_extraction import DictVectorizer
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

df = pd.read_csv('diminutive.train.csv', header=None)

def labeler(pdDF):
    labels = {}
    c = 0  
    for line in (pdDF.iloc[:,-1]):
        if line not in labels:
            labels[line[-1]] = c
            c += 1
    return labels

lst_labels = labeler(df)
print("Labels:", lst_labels)

X_train = df.iloc[:,:-1].to_dict('records')
y_train = []

for label in df.iloc[:,-1]:
    y_train.append(lst_labels[label])

dv = DictVectorizer(sparse=False)
knn = KNeighborsClassifier(n_neighbors=7, metric='manhattan')
p = Pipeline([("dv", dv), ("cls", knn)])

cv_scores_train = cross_val_score(p, X_train, y_train, cv=10)
print('cv_scores mean of train data:{}'.format(np.mean(cv_scores_train)))

X_test_todict = pd.read_csv('diminutive.test.csv', header=None).iloc[:,:-1].to_dict('records')
y_test = pd.read_csv('diminutive.test.csv', header=None).iloc[:,-1]

y_true = []
for label in y_test:
    y_true.append(lst_labels[label])

p.fit(X_train, y_train)
y_pred = p.predict(X_test_todict)

accuracy = accuracy_score(y_true, y_pred)
print('accuracy of test data:{}'.format(np.mean(accuracy)))