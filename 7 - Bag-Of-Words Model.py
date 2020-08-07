"""
Created on Tue Aug 4 2020

@author: Amruth Bellala
@title: NLP Using NLTK

"""


import nltk
import heapq
import re
import numpy as np

paragraph = input("enter the Query : ")
               
               
# Tokenize sentences
data = nltk.sent_tokenize(paragraph)
for i in range(len(data)):
    data[i] = data[i].lower()
    data[i] = re.sub(r'\W',' ',data[i])
    data[i] = re.sub(r'\s+',' ',data[i])


# Creating word histogram
word2count = {}
for i in data:
    words = nltk.word_tokenize(i)
    for word in words:
        if word not in word2count.keys():
            word2count[word] = 1
        else:
            word2count[word] += 1
            
# Selecting best 100 features
frequ_words = heapq.nlargest(100,word2count,key=word2count.get)

# Converting sentences to vectors
X = []
for i in data:
    vector = []
    for word in frequ_words:
        if word in nltk.word_tokenize(i):
            vector.append(1)
        else:
            vector.append(0)
    X.append(vector)
        
X = np.asarray(X)