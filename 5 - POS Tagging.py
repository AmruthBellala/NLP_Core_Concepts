"""
Created on Tue Aug 4 2020

@author: Amruth Bellala
@title: NLP Using NLTK

"""

# Next Step after removing Stop Words

import nltk

paragraph = input("Enter the query: ")
               
               
# POS Tagging
words = nltk.word_tokenize(paragraph)

taged_words = nltk.pos_tag(words)


# Tagged word paragraph
word_tags = []
for tw in taged_words:
    word_tags.append(tw[0]+"_"+tw[1])

taged_paragraph = ' '.join(word_tags)

print(taged_paragraph)