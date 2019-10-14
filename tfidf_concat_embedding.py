# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 01:39:52 2019

@author: ideap
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

#data pre-processing; 
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


df = df.loc[:, ['question_type', 'question_c']]
df['any_fol']=df['question_c'].str.contains('any') & df['question_c'].str.contains('following')  
df['all_app']=df['question_c'].str.contains('all') & (df['question_c'].str.contains('apply') |   
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
x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.3, stratify=y_df, random_state=38104)
del [y_df, x_df]   

#tf-idf
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=None, min_df=.01, max_df=.99, 
                       strip_accents='unicode', norm='l2') 
train_tfidf = vectorizer.fit_transform(x_train['question_c']).todense()
test_tfidf = vectorizer.transform(x_test['question_c']).todense()
col = ['feat_'+i for i in vectorizer.get_feature_names()]     
train_tfidf = pd.DataFrame(train_tfidf, columns=col) 
test_tfidf = pd.DataFrame(test_tfidf, columns=col)

x_train_all = pd.concat([train_tfidf, x_train.reset_index()[['any_fol', 'all_app', 'question_c']]], axis=1)
x_test_all = pd.concat([test_tfidf, x_test.reset_index()[['any_fol', 'all_app', 'question_c']]], axis=1)

#recode labels: integers [0: num_classes] first, then to_categorical();
train_label = utils.to_categorical(y_train, num_classes=y_train.nunique())    
test_label = utils.to_categorical(y_test, num_classes=y_test.nunique())  

class_weights = class_weight.compute_class_weight('balanced', y_train.unique(), y_train)


print('Building model..........')
corpus_size = 8000
sentence_len=50  

aux_input = Input(shape=(len(x_train_all.columns)-1,), name='aux_input') # num of non-NLP features
nlp_input = Input(shape=(sentence_len,), name='nlp_input') # 150 is length of sentence allowed  
emb = Embedding(input_dim = corpus_size, output_dim=256, input_length=sentence_len)(nlp_input) # 256 is dimension of word embedding
nlp_1 = Conv1D(filters=30, kernel_size=(3,), strides=1, padding='same', activation='relu')(emb) # 30 filters of size 3
nlp_2 = Conv1D(filters=32, kernel_size=(6,), strides=1, padding='same', activation='relu')(emb) # 32 filters of size 6
nlp_3 = Conv1D(filters=50, kernel_size=(9,), strides=1, padding='same', activation='relu')(emb) # 50 filters of size 9
nlp_out = keras.layers.concatenate([nlp_1, nlp_2, nlp_3])    
nlp_out = Conv1D(filters=88, kernel_size=(3,), strides=1, padding='valid', activation='relu')(nlp_out)
#nlp_out = MaxPooling1D(pool_size=148)(nlp_out) 
nlp_out = GlobalMaxPooling1D()(nlp_out)
x = keras.layers.concatenate([nlp_out, aux_input])  
x = Dense(256, activation='relu')(x)  
x = Dropout(0.3)(x)  
x = Dense(128, activation='relu')(x)  
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)    
x = Dropout(0.3)(x)
x = Dense(32, activation='relu')(x)  
x = Dropout(0.3)(x)
main_output = Dense(y_train.nunique(), activation='softmax', name='main_output')(x) # 5 output classes 

model = Model(inputs=[nlp_input, aux_input], output=main_output)
model.summary()
model.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')

print('Process the features...........')
tokenizer = Tokenizer(num_words = corpus_size)
tokenizer.fit_on_texts(x_train['question_c'])  
'''
#summarize the tokenizer;
print(tokenizer.word_counts)
print(tokenizer.document_count)
print(tokenizer.word_index)
print(tokenizer.word_docs)
'''  

class_weights = class_weight.compute_class_weight('balanced', y_train.unique(), y_train)

#further prepare the inputs; 
nlp_train = sequence.pad_sequences(tokenizer.texts_to_sequences(x_train_all['question_c']), maxlen=sentence_len) #doc * word index; 
aux_train = x_train_all.iloc[:,:-1].values
nlp_test = sequence.pad_sequences(tokenizer.texts_to_sequences(x_test_all['question_c']), maxlen=sentence_len)
aux_test = x_test_all.iloc[:,:-1].values

#labels need to be integers [0: num_classes] first  
train_label = utils.to_categorical(y_train, num_classes=y_train.nunique())    
test_label = utils.to_categorical(y_test, num_classes=y_test.nunique())    

from numpy.random import seed
seed(392)    
bats=[128]  
pats=[5]    

for b in bats:  
    for p in pats:
        train_df=({'nlp_input': nlp_train, 'aux_input': aux_train}, {'main_output': train_label})
        test_df = ({'nlp_input': nlp_test, 'aux_input': aux_test}, {'main_output': test_label}) 
        
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=3, patience=p)
        model.fit(x=train_df[0], y=train_df[1],
                  epochs=200, # Number of epochs
                  #verbose=1, Print description after each epoch
                  batch_size=b,
                  class_weight=class_weights,  
                  callbacks=[es], 
                  validation_data = test_df)  
        
        
        print('***************Model Fitted on test df*****************') 
        print('batch size: ', b, 'patience: ', p )
        pred_prob = model.predict(test_df[0])
        pred_class = pred_prob.argmax(axis=-1)
        #pred_prob_df = pd.dfFrame(df=pred_prob[0:,:])
        
        print("CNN confusion matrix:\n", confusion_matrix(y_test, pred_class).transpose())
        print("CNN accuracy:\n", accuracy_score(y_test, pred_class))  













