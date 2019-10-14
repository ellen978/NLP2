# -*- coding: utf-8 -*-  
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
import string
from decimal import *
import sys
from nltk.stem import PorterStemmer
from nltk import ne_chunk, pos_tag, word_tokenize
import collections
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

from keras import layers
from keras.layers import Input, Embedding, Dense, Dropout, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Flatten
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
import pandas as pd
import keras.utils.np_utils as utils
import keras  
from keras import regularizers
from sklearn.feature_extraction.text import TfidfVectorizer  
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split  
from nltk.stem.lancaster import LancasterStemmer
from sklearn.metrics import confusion_matrix, accuracy_score    

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

df['outcome'] = df['question_type']
df = df.drop(columns='question_type')
df = df[["outcome", "any_fol", "all_app", "question_c"]]

labels = df['outcome'].unique()
dic = {labels[i]:i for i in range(len(labels))}
y_df = pd.Series(dic[i] for i in df['outcome'])  

x_df = df[["any_fol", "all_app", "question_c"]]

#train-test-prep
x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.3, stratify=y_df, random_state=1234)   

#model 2: CNN; 
print('Building model..........')
corpus_size = 8000
sentence_len=50  

aux_input = Input(shape=(len(x_train.columns)-1,), name='aux_input') # num of non-NLP features
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
x = Dense(64, activation='relu')(x)  #, kernel_regularizer=regularizers.l2(0.01)
x = Dropout(0.3)(x)
x = Dense(32, activation='relu')(x)

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
nlp_train = sequence.pad_sequences(tokenizer.texts_to_sequences(x_train['question_c']), maxlen=sentence_len) #doc * word index; 
aux_train = x_train.iloc[:,:2].values
nlp_test = sequence.pad_sequences(tokenizer.texts_to_sequences(x_test['question_c']), maxlen=sentence_len)
aux_test = x_test.iloc[:,:2].values

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
        

'''
# plot training history
from matplotlib import pyplot
pyplot.plot(model['loss'], label='train')
pyplot.plot(model.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
'''

 
''' 
patience=15, batch_size=128
 [[141   4  27   1   5   3   0   0]
 [  1  64   9   1   1   1   0   2]
 [  9   9 135   7   7   1   2   0]
 [  2   2   4  73   1   1   1   0]
 [  3   1   6   2  21   0   0   0]
 [  1   0   3   1   2  28   0   0]
 [  0   0   0   0   0   0   1   0]
 [  0   0   0   0   0   0   0   3]]
CNN accuracy:
 0.7952218430034129
 
#+ 1 NLP
CNN confusion matrix:
 [[134   4  23   3   5   1   0   2]
 [  1  58  10   1   1   2   0   0]
 [ 13  12 137   6  13   1   3   0]
 [  2   1   1  72   1   1   0   0]
 [  7   3  13   2  15   0   0   0]
 [  0   2   0   1   2  29   0   0]
 [  0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   1   3]]
CNN accuracy:
 0.764505119453925
 
 #- 1 NLP (filter=9)
batch size:  128 patience:  15
CNN confusion matrix:
 [[139   7  25   1   9   1   0   0]
 [  2  65  11   0   3   2   0   0]
 [ 10   5 137   3   7   1   3   0]
 [  2   3   3  80   0   0   1   0]
 [  4   0   7   0  18   1   0   2]
 [  0   0   0   1   0  29   0   0]
 [  0   0   0   0   0   0   0   0]
 [  0   0   1   0   0   0   0   3]]
CNN accuracy:
 0.8037542662116041
  
#-1 NLP (filter=9), 1 more dense layer (16);   
batch size:  128 patience:  15
CNN confusion matrix:
 [[135   4  26   4   8   0   0   0]
 [  0  66  10   1   1   0   1   0]
 [ 11   6 139   3   9   2   2   2]
 [  7   3   3  75   1   1   0   0]
 [  4   0   4   0  18   1   0   0]
 [  0   0   2   2   0  30   1   0]
 [  0   0   0   0   0   0   0   0]
 [  0   1   0   0   0   0   0   3]]
CNN accuracy:
 0.7952218430034129

#64 w/ dropout rate=0.2
batch size:  128 patience:  15
CNN confusion matrix:
 [[141   3  22   3   7   0   0   2]
 [  2  69  15   1   0   2   1   0]
 [ 10   6 142   5  11   2   3   0]
 [  1   1   0  71   1   1   0   0]
 [  3   0   4   1  18   1   0   0]
 [  0   1   0   4   0  28   0   0]
 [  0   0   0   0   0   0   0   0]
 [  0   0   1   0   0   0   0   3]]
CNN accuracy:
 0.8054607508532423  
 
#add l2 regularizer, 0.01 on dense layer 64; 
CNN confusion matrix:
 [[142   2  18   3   7   0   1   0]
 [  0  66   9   1   0   1   0   0]
 [ 11  10 150   7  12   3   3   2]
 [  1   1   1  71   1   1   0   0]
 [  3   0   4   1  17   1   0   0]
 [  0   1   1   2   0  28   0   0]
 [  0   0   0   0   0   0   0   0]
 [  0   0   1   0   0   0   0   3]]
CNN accuracy:
 0.8139931740614335

##To do: to tune the structure (number of NLP filters, number of dense layers)

'''







