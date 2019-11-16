import numpy as np
import pandas as pd

from keras.metrics import categorical_accuracy
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

df_test = pd.read_csv('vacatures_test_labels.csv', header=0, index_col=False)
df_true = pd.read_csv('vacatures_test_metlabels.csv', header=0)

y_test = df_test['type']
y_true = df_true['type']

print(accuracy_score(y_true, y_test))