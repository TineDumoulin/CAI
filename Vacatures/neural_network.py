import keras 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from imblearn.over_sampling import RandomOverSampler
from IPython.core.display import display, HTML
from keras import regularizers
from keras.callbacks import Callback, ModelCheckpoint
from keras.layers import Input, Dense, Dropout, Embedding, LSTM, Flatten
from keras.models import Model, Sequential, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from langdetect import detect
from scipy.stats import randint as sp_randint
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from time import time

df = pd.read_csv('vacatures_train.csv', header=0)
# delete instance with strange labels ([if !supportLists], [endif])
df.drop(df.index[578])

""" # detect the language of the instance and add language labels to the dataframe
lang_list = []
print('Detecting languages of instances...\n')
for description in df['description']:
    lang = detect(description)
    lang_list.append(lang)

pd_series_langs = pd.Series(lang_list)
df = df.assign(language=pd_series_langs)
df = df[df.language == 'nl']
df.to_csv('vacatures_train_dutch.csv') """

# preprocessing the job descriptions
df = pd.read_csv('vacatures_train_dutch.csv', header=0)

""" print(df.head())

checking the class distribution
print(df['type'].value_counts()) """

# calculate the number of words
df['num_words'] = df['description'].apply(lambda x : len(x.split()))

# binning the posts by word count
bins = [0,50,75, np.inf]
df['bins'] = pd.cut(df.num_words, bins=[0,50,100,200,300, np.inf], labels=['0-50', '50-100', '100-200','200-300' ,'>300'])
word_distribution = df.groupby('bins').size().reset_index().rename(columns={0:'counts'})
# print(word_distribution.head())

# set number of classes and target variable and converting tags
num_class = len(np.unique(df.type.values))
X = df.description.values

encoder = LabelEncoder()
y = encoder.fit_transform(df['type'])

# tokenize the input
MAX_LENGTH = 250
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
description_seq = tokenizer.texts_to_sequences(X)
description_seq_padded = pad_sequences(description_seq, padding='pre', maxlen=MAX_LENGTH)

X_train, X_val, y_train, y_val = train_test_split(description_seq_padded, y, test_size=0.2, random_state=277)

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
embedding_layer = Embedding(vocab_size, 32, input_length=MAX_LENGTH)(inputs)
x = Flatten()(embedding_layer)
x = (Dropout(0.5))(x)
# x = Dense(24, kernel_regularizer=regularizers.l2(0.001), activation='relu')(x)
## x = Dense(24, kernel_regularizer=regularizers.l2(0.001), activation='relu')(x)
predictions = Dense(num_class, kernel_regularizer=regularizers.l2(0.001), activation='softmax')(x)

model = Model(inputs=[inputs], outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

model.summary()

# fitting the model on the train set and saving the best model
filepath="neural_net.hdf5"
checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
history = model.fit(X_resampled, y=to_categorical(y_resampled), batch_size=64, epochs=50, callbacks=[checkpointer], validation_split=0.25, shuffle=True, verbose=1)

# plotting the train and validation loss to diagnose over- or underfitting
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Validation loss')

plt.title('Training and validation loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()

plt.show()

# plotting the train and validation accuracy
plt.clf()

plt.plot(history.history['acc'], label='Training accuracy')
plt.plot(history.history['val_acc'], label='Validation accuracy')

plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# looking at accuracy
predicted = model.predict(X_val)
predicted = np.argmax(predicted, axis=1)
print('Test accuracy: ', accuracy_score(y_val, predicted)) """





# TEST DATA PREDICTION
# opening the test file
df_test = pd.read_csv('vacatures_test_zonderlabel.csv', header=0)

# Tokenizing (and preprocessing) the test data
description_seq_test = tokenizer.texts_to_sequences(df_test['description'])
X_test = pad_sequences(description_seq, padding='pre', maxlen=MAX_LENGTH)

# loading the model
print("Creating model and loading weights from file...\n")
NN = load_model("neural_net.h5")
NN.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
print("Created model and loaded weights from file")

# predicting labels on test data
y_test = NN.predict(X_test)
y_test = np.argmax(y_test, axis=1)

# decoding the labels
y_test_decoded = encoder.inverse_transform(y_test)
print(df_test[:50:])
print(type(y_test_decoded))
print(y_test_decoded.shape)
print(y_test_decoded[:50:])

# saving the labels to CSV
np.savetxt("vacatures_test_labels_DumoulinTine.csv", y_test_decoded, delimiter=",")