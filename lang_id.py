import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('langid.csv')
X = df.iloc[:,0]
y = df.iloc[:,1]

print(X.shape)
print(len(y)

#def labeler(pdDF):
#    labels = {}
#    c = 0  
#    for line in (pdDF.iloc[:,1]):
#        if line not in labels:
#            labels[line[-3:]] = c
#            c += 1
#    return labels

#lbls = labeler(df)

#y_labels = []
#for label in y[:12]:
#    y_labels.append(lbls[label])

#print(y_labels[:10])
# tfidf = TfidfVectorizer()

knn = KNeighborsClassifier()
params = {'n_neighbors': [3, 5, 7], 'metric': ['manhattan', 'euclidian', 'chebyshev'], 'lowercase': [True, False]}
pipe = Pipeline([("tfidf", tfidf), ("cls", knn)])

search = GridSearchCV(pipe, params, iid=False, cv=10, return_train_score=False)
#search.fit(X_tr, y_tr)

print('Best parameter (CV score=%0.3f):' % search.best_score_)
print(search.best_params_)