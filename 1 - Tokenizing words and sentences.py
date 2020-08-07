"""
Created on Tue Aug 4 2020

@author: Amruth Bellala
@title: NLP Using NLTK

"""

# Tokenization of paragraphs/sentences
import nltk

paragraph = input("Enter the Query: ")
           
# Tokenizing sentences
sentences = nltk.sent_tokenize(paragraph)
print(sentences)

# Tokenizing words
words = nltk.word_tokenize(paragraph)
print(words)
               