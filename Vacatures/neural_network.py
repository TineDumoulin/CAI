import keras 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import RandomOverSampler
from IPython.core.display import display, HTML
from keras.callbacks import Callback, ModelCheckpoint
from keras.layers import Input, Dense, Dropout, Embedding, LSTM, Flatten
from keras.models import Model, Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from scipy.stats import randint as sp_randint
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from time import time

# reading the data
df = pd.read_csv('vacatures_train_dutch.csv', header=0)
# print(df.head())

# checking the class distribution
# print(df['type'].value_counts())

# calculate the number of words
df['num_words'] = df['description'].apply(lambda x : len(x.split()))

# binning the posts by word count
bins = [0,50,75, np.inf]
df['bins'] = pd.cut(df.num_words, bins=[0,50,100,200,300, np.inf], labels=['0-50', '50-100', '100-200','200-300' ,'>300'])
word_distribution = df.groupby('bins').size().reset_index().rename(columns={0:'counts'})
# print(word_distribution.head())

# converting tags
df['type'] = df.type.astype('category').cat.codes

# set number of classes and target variable
num_class = len(np.unique(df.type.values))
X = df.description.values
y = df.type.astype('category').cat.codes

# tokenize the input
MAX_LENGTH = 200
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
description_seq = tokenizer.texts_to_sequences(X)
description_seq_padded = pad_sequences(description_seq, padding='pre', maxlen=MAX_LENGTH)

X_train, X_test, y_train, y_test = train_test_split(description_seq_padded, y, test_size=0.2, random_state=277)

# oversampling the rare classes
ros = RandomOverSampler(random_state=277)
ros.fit(X_train, y_train)
X_resampled, y_resampled = ros.sample(X_train, y_train)
unique, counts = np.unique(y_resampled, return_counts=True)
print(np.asarray((unique, counts)).T)
print(X_resampled.shape, y_resampled.shape)

vocab_size = len(tokenizer.word_index) + 1

""" # simple deep learning model
inputs = Input(shape=(MAX_LENGTH, ))
embedding_layer = Embedding(vocab_size, 128, input_length=MAX_LENGTH)(inputs)
x = Flatten()(embedding_layer)
x = (Dropout(0.6))(x)
x = Dense(32, activation='relu')(x)
predictions = Dense(num_class, activation='softmax')(x)

model = Model(inputs=[inputs], outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

model.summary()

filepath="weights-simple.hdf5"
checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
history = model.fit([X_resampled], batch_size=64, y=to_categorical(y_resampled), verbose=1, validation_split=0.25, shuffle=True, epochs=3, callbacks=[checkpointer])

# understanding the model fit
df2 = pd.DataFrame({'epochs':history.epoch, 'accuracy': history.history['acc'], 'validation_accuracy': history.history['val_acc']})
g = sns.pointplot(x="epochs", y="accuracy", data=df2, fit_reg=False)
g = sns.pointplot(x="epochs", y="validation_accuracy", data=df2, fit_reg=False, color='green')

# looking at accuracy
predicted = model.predict(X_test)
predicted = np.argmax(predicted, axis=1)

print('f1: 4', f1_score(y_test, predicted, average='weighted'))
print('recall: ', recall_score(y_test, predicted, average='weighted'))
print('precision: ', precision_score(y_test, predicted, average='weighted'))

# plotting the train and validation loss to diagnose over- or underfitting
plt.plot(history.history['loss'], label='model train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.axis('equal')
plt.title('model train loss vs. validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show() """

# recurrent neural network
inputs = Input(shape=(MAX_LENGTH, ))
embedding_layer = Embedding(vocab_size, 128, input_length=MAX_LENGTH)(inputs)
x = Flatten()(embedding_layer)
x = (Dropout(0.5))(x)
x = LSTM(64)(embedding_layer)
x = Dense(32, activation='relu')(x)
x = (Dropout(0.5))(x)
predictions = Dense(num_class, activation='softmax')(x)
model = Model(inputs=[inputs], outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc']) 

model.summary()

filepath="weights.hdf5"
checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

history = model.fit([X_resampled], batch_size=64, y=to_categorical(y_resampled), verbose=1, validation_split=0.25, shuffle=True, epochs=15, callbacks=[checkpointer])

df3 = pd.DataFrame({'epochs':history.epoch, 'accuracy': history.history['acc'], 'validation_accuracy': history.history['val_acc']})
g = sns.pointplot(x="epochs", y="accuracy", data=df3, fit_reg=False)
g = sns.pointplot(x="epochs", y="validation_accuracy", data=df3, fit_reg=False, color='green')

# plotting the train and validation loss to diagnose over- or underfitting
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('train loss vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

# looking at accuracy
model.load_weights('weights.hdf5')
predicted = model.predict(X_test)
predicted = np.argmax(predicted, axis=1)

print('accuracy: ', accuracy_score(y_test, predicted))
print('f1: 4', f1_score(y_test, predicted, average='weighted'))
print('recall: ', recall_score(y_test, predicted, average='weighted'))
print('precision: ', precision_score(y_test, predicted, average='weighted'))

""" # recurrent NN with RandomizedSearchCV
def create_model():
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=MAX_LENGTH))
    model.add(LSTM(64))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_class, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    return model

model = KerasClassifier(build_fn=create_model())
param_distr = {'epochs': sp_randint(10, 300), 'batch_size': sp_randint(20, 100)}
n_iter_search = 30

rdms_cv = RandomizedSearchCV(estimator=model, param_distributions=param_distr, n_iter=n_iter_search, n_jobs=-1, iid=False, cv=3, verbose=3, random_state=277, )
start = time()
rdms_cv_result = rdms_cv.fit(X_resampled, y_resampled)

print("RandomizedSearchCV took %.2f seconds for %d candidates parameter settings." % ((time() - start), n_iter_search))

# summarize best results
print("Best: %f using %s" % (rdms_cv_result.best_score_, rdms_cv_result.best_params_))
means = rdms_cv_result.cv_results_['mean_test_score']
stds = rdms_cv_result.cv_results_['std_test_score']
params = rdms_cv_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param)) """