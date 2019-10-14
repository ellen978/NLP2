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
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
from nltk.corpus import stopwords, wordnet
from nltk import PorterStemmer, WordNetLemmatizer  
import string
from decimal import *
import sys
from nltk.stem import PorterStemmer
from nltk import ne_chunk, pos_tag, word_tokenize
import collections
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora, models, similarities
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel  
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', 500)

data=pd.read_csv('takehome_ml_data.csv')
data.head(10)
data['question_type'].value_counts() #8 question types; 

data['question_c'] = data['question'].fillna('')
data['question_c']=data['question_c'].str.lower()
data['question_c']=data['question_c'].str.split()
data['question_c']=data['question_c'].apply(lambda x: [re.sub('[^a-zA-Z]*', '', i) for i in x])
data['question_c']=data['question_c'].apply(lambda x: [c for c in x if not c.isdigit()])


a={'what', 'which', 'who', 'whom', 'or', 'above', 'below', 'when', 'where', 'why', 'how', 'all', 'any', 
'both', 'each', 'more', 'most', 'no', 'nor', 'not', 'only', 'than', 'very'} 
b={'would'}         
c=set(stopwords.words('english'))
sw=c|b 
sw=sw-a
data['question_c']=data['question_c'].apply(lambda x: ' '.join([c for c in x if c not in sw]))

def dfdic(text):    
    dic={}  
    for i in text: 
        for j in set(i.split()):
            if j in dic:
                dic[j]+=1
            else:    
                dic[j]=1  
    return dic  
dic=dfdic(data['question_c'])  
sw2={k for k, v in dic.items() if len(k)==1 or v==1}  
data['question_c']=data['question_c'].apply(lambda x: ' '.join([c for c in x.split() if c not in sw2]))

'''
rank=13  #top 10 words;     
res={}
for c in data['question_type'].unique():
    dic=dfdic(data.loc[data['question_type']==c, 'question_c'])
    line=sorted(dic.values(), reverse=True)[rank:(rank+1)][0]   
    keywrd=[k for k, v in dic.items() if v>line]
    res[c]=keywrd    
res
'''

data=data.loc[:, ['question_c', 'question_type', 'question']]
data['any_fol']=data['question_c'].str.contains('any') & data['question_c'].str.contains('following')  
data['all_app']=data['question_c'].str.contains('all') & (data['question_c'].str.contains('apply') | data['question_c'].str.contains('select')) 


#train-test-prep & tfidf
Y=data[['question_type', 'any_fol', 'all_app']]  
X=data['question_c']  

rs=9243  
x_train, x_test, z_train, z_test = train_test_split(X, Y, test_size=0.25, random_state=rs)     
ytrain=z_train['question_type']  
ytest=z_test['question_type']

tfidfv=TfidfVectorizer(min_df=1, ngram_range=(1, 2), stop_words=None, max_features=10000, 
                       strip_accents='unicode', norm='l2')    
xtrain=tfidfv.fit_transform(x_train).todense()
xtest=tfidfv.transform(x_test).todense()  

col = ['feat_'+i for i in tfidfv.get_feature_names()]     
xtrain = pd.DataFrame(xtrain, columns=col)    
xtest = pd.DataFrame(xtest, columns=col) 

ztrain=pd.DataFrame(z_train.drop('question_type', axis=1)).reset_index(drop=True)
ztest=pd.DataFrame(z_test.drop('question_type', axis=1)).reset_index(drop=True)

xtrain=pd.concat([xtrain, ztrain], axis=1)
xtest=pd.concat([xtest, ztest], axis=1)


import os  #to solve mac-specific problem of running lightgbm 
os.environ['KMP_DUPLICATE_LIB_OK']='True'

##model 1: lightgbm
import lightgbm

#recode labels to 0-count to fit for lightgbm model (RF no need)
labels=['single-select', 'textarea', 'rate', 'description', 'multi-select', 'matrix-likert', 
        'number', 'rank']
ldic={}
for i, c in enumerate(labels): 
    ldic[c]=i
ytrain=pd.Series([ldic[i] for i in ytrain])
ytest=pd.Series([ldic[i] for i in ytest])  

#data prep; 
d_train=lightgbm.Dataset(xtrain, label=ytrain)
d_test=lightgbm.Dataset(xtest, label=ytest)
nb_classes=len(ytrain.value_counts()) 

params={'boosting_type':'gbdt', 'objective':'multiclass', 'metric':'multi_logloss', 'learning_rate':0.005,
       'num_classes':nb_classes, 'subsample_freq':3, 'min_child_samples':5, 'min_child_weight':0.1,
       'colsample_bytree':0.8, 'subsample':0.7, 'min_split_gain':0.05, 'max_bin':25, 'max_depth':-1,
       'num_leaves':25, 'early_stopping_round':100, 'random_state':rs}   
n_estimators=3000
watchlist=[d_train, d_test]
model=lightgbm.train(params=params, train_set=d_train, num_boost_round=n_estimators, valid_sets=watchlist, verbose_eval=10)

#lightgbm predict;   
y_pred_prob = model.predict(xtest, num_iteration=model.best_iteration) 
y_pred_class = y_pred_prob.argmax(axis=-1)
y_label=list(ytest)
y_pred=list(y_pred_class)

y_pred_prob = pd.DataFrame(y_pred_prob)
y_prob = pd.DataFrame(y_pred_prob)

print("LGB cv confusion matrix:\n",  "\n", confusion_matrix(y_label, y_pred).transpose())
print("LGB cv accuracy:\n", "\n", accuracy_score(y_label, y_pred))  




