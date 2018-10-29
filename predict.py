from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import load_model
import pickle
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
lmtzr = WordNetLemmatizer()


with open('Input_text.txt', 'r') as f1:
   input_txt = f1.readlines()

for i in range(len(input_txt)):
    input_words = input_txt[i].split()
    
    def onehot(ltr):
        return [1 if i==ord(ltr) else 0 for i in range(97,123)]
    
    def onehotvec(s):
        return [onehot(c) for c in list(s.lower())]
    
    
    def padd(vec_word):
        for i in range(len(vec_word),15):
            vec_word.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        return vec_word
    
    def trim(vec_word):
        trimmed_word=[]
        for i in range(0,15):
            trimmed_word.append(vec_word[i])
        return trimmed_word
    
    lemmatized_input_words = []
    
    for word in input_words:
        lemmatized_input_words.append(lmtzr.lemmatize(word))
    corpora_char=[]
    corpora_string=""
    vectorized_test_corpora=[]
    i=0
    j=0
    
    for word in lemmatized_input_words:
        #print(i," ",word)
        for c in list(word):
            if(ord(c) in range(97,123)):
                corpora_char.append(c)
        corpora_string=''.join(corpora_char)
        vectorized_test_corpora.append(onehotvec(corpora_string))
            
        if(len(vectorized_test_corpora[j])<15):
            vectorized_test_corpora[j] = padd(vectorized_test_corpora[j])
        if(len(vectorized_test_corpora[j])>15):    
            vectorized_test_corpora[j] =trim(vectorized_test_corpora[j])
        #print(vectorized_test_corpora[j],"\n")
        corpora_char=[]
        corpora_string=""
        j=j+1
        i=i+1
    
    Xtest=np.array(vectorized_test_corpora)
    Xtest=Xtest.reshape(len(vectorized_test_corpora), 15, 26,1)
    
    model = load_model('trained_model.h5')
    
    Ytest_sentiment = model.predict_classes(Xtest)
    
    Ytest_sentiment =Ytest_sentiment.flatten()
    Ytest_sentiment =Ytest_sentiment.tolist()
    
    print("\n\n")
    
    for i in range(len(input_words)):
        if(Ytest_sentiment[i]==1):
            print("\033[94m"+input_words[i]+"\033[0m", end=" ")
        else:
            print(input_words[i], end=" ")
    
    print("\n")
    
    
    