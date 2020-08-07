"""
Created on Tue Aug 4 2020

@author: Amruth Bellala
@title: NLP Using NLTK

"""

# Next step after performing Tokenization of paragraphs/sentences

import nltk
from nltk.stem import PorterStemmer

paragraph =input("Enter the query: ")
               
               
sentences = nltk.sent_tokenize(paragraph)
stemming = PorterStemmer()


# Stemming
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [stemming.stem(word) for word in words]
    sentences[i] = ' '.join(words)        

print(sentences)       