"""
Created on Tue Aug 4 2020

@author: Amruth Bellala
@title: NLP Using NLTK

"""

import nltk

paragraph = input("Enter the query: ")
               
               
# POS Tagging
words = nltk.word_tokenize(paragraph)
taged_words = nltk.pos_tag(words)

# Named entity recognition
namedEntity = nltk.ne_chunk(taged_words)
namedEntity.draw()