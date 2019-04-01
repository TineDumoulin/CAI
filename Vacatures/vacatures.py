import pandas as pd
from langdetect import detect
from googletrans import Translator
import collections

df = pd.read_csv('vacatures_train.csv', header=0)
df.drop(df.index[578]) # because of the weird labels ([if !supportLists], [endif])
X = df['description']
y = df['type']

lang_lbls = {}
lang_list = []

print('Detecting languages of instances...\n')
for description in X[:100]:
    lang = detect(description)

    lang_lbls[description] = lang
    lang_list.append(lang)

translations = []

print('Translating non-Dutch instances...\n')
for item, lang in lang_lbls.items():
    translator = Translator()
    if lang == 'nl':
        translations.append(item)
    else:
        item = str(translator.translate(item, dest='nl').text)
        translations.append(item)
        
pd_series_langs = pd.Series(lang_list)
pd_series_translations = pd.Series(translations)

df = df.assign(translated_description=pd_series_translations, lang_description=pd_series_langs)
print(df[:0])
df.to_csv('vacatures_train_translated.csv')