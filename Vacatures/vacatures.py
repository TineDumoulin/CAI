import pandas as pd
import numpy as np
from langdetect import detect
import collections
import string
import re
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
import sklearn.datasets as skds
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from pathlib import Path

df = pd.read_csv('vacatures_train.csv', header=0)
# delete instance with strange labels ([if !supportLists], [endif])
df.drop(df.index[578])

""" lang_list = []
# detect the language of the instance
print('Detecting languages of instances...\n')
for description in df['description']:
    lang = detect(description)
    lang_list.append(lang)

# add the language label as a column to the dataframe and export it to a new csv-file (so I don't have to run this part each time)
pd_series_langs = pd.Series(lang_list)
df = df.assign(language=pd_series_langs)
df = df[df.language == 'nl']
df.to_csv('vacatures_train_dutch.csv') """

# preprocessing the job descriptions
df = pd.read_csv('vacatures_train_dutch.csv', header=0)
preprocessed_text = []

for line in df['description']:
    # delete punctuation, whitespace, convert to lowercase
    line = line.translate(line.maketrans('','', string.punctuation)).strip().lower() 
    # delete numbers
    line = re.sub(r'\d+', '', line) 
    # delete double spaces
    line = re.sub(' +', ' ', line) 
    # append to list of preprocessed job descriptions
    preprocessed_text.append(line)

# replace the original job descriptions in the dataframe with the preprocessed text
df['description'] = pd.Series(preprocessed_text)

# assign instances and labels and create a train/test split
X = df['description']
y = df['type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=2707)

unique_labels = y_train.unique().tolist()
num_labels = len(unique_labels)
vocab_size = 15000
batch_size = 100
 
# label encode the target variable 
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)

# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(X_train)
X_train_tfidf = tfidf_vect.transform(X_train)
X_test_tfidf = tfidf_vect.transform(X_test)

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return accuracy_score(predictions, y_test)

def create_model_architecture(input_size):
    # create input layer 
    input_layer = layers.Input((input_size, ), sparse=True)
    
    # create hidden layer
    hidden_layer = layers.Dense(100, activation="relu")(input_layer)
    
    # create output layer
    output_layer = layers.Dense(1, activation="softmax")(hidden_layer)

    classifier = models.Model(inputs = input_layer, outputs = output_layer)
    classifier.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy')
    return classifier

classifier = create_model_architecture(X_train_tfidf.shape[1])
accuracy = train_model(classifier, X_train_tfidf, y_train, X_test_tfidf, is_neural_net=True)
print("NN, word Level TF IDF Vectors",  accuracy) 