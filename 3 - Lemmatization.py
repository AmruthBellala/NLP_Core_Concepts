"""
Created on Tue Aug 4 2020

@author: Amruth Bellala
@title: NLP Using NLTK

"""

# Next Step after Tokenization of paragraphs/sentences

import nltk
from nltk.stem import WordNetLemmatizer

paragraph = input("Enter the query: ")
               
               
sentences = nltk.sent_tokenize(paragraph)
lemmatizer = WordNetLemmatizer()

# Lemmatization
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [lemmatizer.lemmatize(word) for word in words]
    sentences[i] = ' '.join(words)               
    
print(sentences)