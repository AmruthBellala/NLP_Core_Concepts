"""
Created on Tue Aug 4 2020

@author: Amruth Bellala
@title: NLP Using NLTK

"""

# Next step after Performing Lemmatization

import nltk
from nltk.corpus import stopwords

paragraph = input("Enter the query: ")
               
# Tokenization            
sentences = nltk.sent_tokenize(paragraph)

# Removing stopwords
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [word for word in words if word not in stopwords.words('english')]
    sentences[i] = ' '.join(words)               

print(sentences)