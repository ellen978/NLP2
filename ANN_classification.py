# -*- coding: utf-8 -*-  
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import re   
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import string
import collections
import nltk  
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('maxent_ne_chunker')
#nltk.download('words')
from nltk.corpus import stopwords, wordnet  
from nltk import PorterStemmer, WordNetLemmatizer  
from nltk import ne_chunk, pos_tag, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from keras import layers
from keras.models import Sequential
from keras.layers import Input, Embedding, Activation, Dense, Dropout, Conv1D, MaxPooling1D, \
GlobalMaxPooling1D, Flatten
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
import keras.utils.np_utils as utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from nltk.stem.lancaster import LancasterStemmer
from keras.optimizers import Adam

pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', 500)

import os
os.chdir('C:\\Users\\ideap\\Desktop\\Jing_project\\project')
print(os.getcwd())  

df = pd.read_csv('survey_data.csv')
df['question_type'].value_counts() #8 question types;
df = df.iloc[:,1:]

df['question_c'] = df['question'].fillna('')
df['question_c']=df['question_c'].str.lower()
df['question_c']=df['question_c'].str.split()  
df['question_c']=df['question_c'].apply(lambda x: [re.sub('[^a-zA-Z]*', '', i) for i in x])
df['question_c']=df['question_c'].apply(lambda x: [c for c in x if not c.isdigit()])

a={'what', 'which', 'who', 'whom', 'or', 'above', 'below', 'when', 'where', 'why', 'how', 'all', 'any', 
'both', 'each', 'more', 'most', 'no', 'nor', 'not', 'only', 'than', 'very'} 
b={'would'}         
c=set(stopwords.words('english'))
sw=c|b 
sw=sw-a
df['question_c']=df['question_c'].apply(lambda x: ' '.join([c for c in x if c not in sw]))

def data_dic(text):    
    dic={}  
    for i in text: 
        for j in set(i.split()):
            if j in dic:
                dic[j]+=1  
            else:    
                dic[j]=1  
    return dic  
dic=data_dic(df['question_c'])  
sw2={k for k, v in dic.items() if len(k)==1 or v==1}  
df['question_c']=df['question_c'].apply(lambda x: ' '.join([c for c in x.split() if c not in sw2]))

'''
rank=13  #top 10 words;     
res={}
for c in df['question_type'].unique():
    dic=dfdic(df.loc[df['question_type']==c, 'question_c'])
    line=sorted(dic.values(), reverse=True)[rank:(rank+1)][0]   
    keywrd=[k for k, v in dic.items() if v>line]
    res[c]=keywrd      
res
'''

df = df.loc[:, ['question_type', 'question_c']]
df['any_fol']=df['question_c'].str.contains('any') & df['question_c'].str.contains('following')  
df['all_app']=df['question_c'].str.contains('all') & (df['question_c'].str.contains('apply') | \
df['question_c'].str.contains('select'))

df["any_fol"] = np.where(df["any_fol"]==True, 1, 0)
df["all_app"] = np.where(df["all_app"]==True, 1, 0)


#model 2: tf-idf + ANN;   
df['outcome'] = df['question_type']
df = df.drop(columns='question_type')
df = df[["outcome", "any_fol", "all_app", "question_c"]]

labels = df['outcome'].unique()
dic = {labels[i]:i for i in range(len(labels))}
y_df = pd.Series(dic[i] for i in df['outcome'])
x_df = df[["any_fol", "all_app", "question_c"]]

#train-test-prep
x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.3, stratify=y_df, random_state=1234)
del [y_df, x_df]   

#tf-idf
vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2), stop_words=None, max_features=5000, 
                       strip_accents='unicode', norm='l2') 
train_tfidf = vectorizer.fit_transform(x_train['question_c']).todense()
test_tfidf = vectorizer.transform(x_test['question_c']).todense()
col = ['feat_'+i for i in vectorizer.get_feature_names()]     
train_tfidf = pd.DataFrame(train_tfidf, columns=col) 
test_tfidf = pd.DataFrame(test_tfidf, columns=col)

x_train_all = pd.concat([train_tfidf, x_train.reset_index()[['any_fol', 'all_app']]], axis=1)
x_test_all = pd.concat([test_tfidf, x_test.reset_index()[['any_fol', 'all_app']]], axis=1)

#recode labels: integers [0: num_classes] first, then to_categorical();
train_label = utils.to_categorical(y_train, num_classes=y_train.nunique())    
test_label = utils.to_categorical(y_test, num_classes=y_test.nunique())  

class_weights = class_weight.compute_class_weight('balanced', y_train.unique(), y_train)

print('Building ANN model based on tfidf + 2 cols..........')    
# define training
model = Sequential()
model.add(Dense(1000, input_dim = len(x_train_all.columns)))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(y_train.nunique()))
model.add(Activation('softmax'))  

optimizer = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=optimizer, metrics=['accuracy'], loss='categorical_crossentropy')
print(model.summary())

#class_weights = class_weight.compute_class_weight('balanced', ytrain.unique(), ytrain)
model.fit(x_train_all, train_label, validation_data = (x_test_all, test_label), 
          class_weight=class_weights, epochs=10, batch_size=64, verbose=1) 
#class_weight=class_weights
     
# predict
pred_prob = model.predict(x_test_all)
pred_class = pred_prob.argmax(axis=-1)

print("ANN test confusion matrix:\n", confusion_matrix(y_test, pred_class).transpose())
print("ANN test accuracy:", accuracy_score(y_test, pred_class))
print("ANN classification report:\n", classification_report(y_test, pred_class))  




