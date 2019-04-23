import keras 
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Dropout, Embedding, LSTM, Flatten
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from IPython.core.display import display, HTML
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

# reading the data
df = pd.read_csv('vacatures_train_dutch.csv', header=0)
print(df.head())

# checking the class distribution
print(df['type'].value_counts())

# converting tags
df['type'] = df.type.astype('category').cat.codes

# calculate the number of words
df['num_words'] = df['description'].apply(lambda x : len(x.split()))

# binning the posts by word count
bins = [0,50,75, np.inf]
df['bins'] = pd.cut(df.num_words, bins=[0,50,100,200,300, np.inf], labels=['0-50', '50-100', '100-200','200-300' ,'>300'])
word_distribution = df.groupby('bins').size().reset_index().rename(columns={0:'counts'})
print(word_distribution.head())

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

vocab_size = len(tokenizer.word_index) + 1

# simple deep learning model
inputs = Input(shape=(MAX_LENGTH, ))
embedding_layer = Embedding(vocab_size, 128, input_length=MAX_LENGTH)(inputs)
x = Flatten()(embedding_layer)
x = (Dropout(0.5))(x)
x = Dense(32, activation='relu')(x)

predictions = Dense(num_class, activation='softmax')(x)
model = Model(inputs=[inputs], outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

model.summary()
filepath="weights-simple.hdf5"
checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
history = model.fit([X_resampled], batch_size=64, y=to_categorical(y_resampled), verbose=1, validation_split=0.25, shuffle=True, epochs=10, callbacks=[checkpointer])

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

# recurrent neural network
inputs = Input(shape=(MAX_LENGTH, ))
embedding_layer = Embedding(vocab_size, 128, input_length=MAX_LENGTH)(inputs)

x = LSTM(64)(embedding_layer)
x = (Dropout(0.5))(x)
x = Dense(32, activation='relu')(x)
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

model.load_weights('weights.hdf5')
predicted = model.predict(X_test)
predicted = np.argmax(predicted, axis=1)

print('f1: 4', f1_score(y_test, predicted, average='weighted'))
print('recall: ', recall_score(y_test, predicted, average='weighted'))
print('precision: ', precision_score(y_test, predicted, average='weighted'))