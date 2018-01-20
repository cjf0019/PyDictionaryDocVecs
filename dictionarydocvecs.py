# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 08:40:59 2017
Takes a pretrained Doc2VecWordFixed model and trains docvecs based on PyDictionary
definitions. The definitional word is taken to be the doctag, and the word vectors
themselves are fixed from previous training. 
@author: InfiniteJest
"""

import numpy as np
import re
import os
import nltk
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
stops = set(nltk.corpus.stopwords.words("english"))
stops.update(['.',',','"',"'", '?', '!', ':', ';','(',')','[',']','{','}',"''",'``'])
import gensim

os.chdir('C:\\Users\\InfiniteJest\\Documents\\Python_Scripts')
import doc2vecwordfixed


wikimodel = doc2vecwordfixed.Doc2VecWordFixed.load('wiki100dmnolbls001samp3000mc')
#vocabulary = wikimodel.vocab.keys()
vocabfile = open('dictionaryvocabwords', 'r')
vocabfile = vocabfile.read()
vocabulary = vocabfile.split('\n')
vocabulary.sort()
vocabulary.pop(0)


def review_sentence(sentence):
    """
    SPECIAL NOTE:
    Could mess with x5 if wanting to keep each definition corresponding to the same word separate
    """
    x = re.sub("\'s", '', sentence.lower())
    x2 = re.sub('^\$\d*$', 'money', x)
    x3 = re.sub('^1\d\d\d$', 'year', x2)
    x4 = re.sub('^\d+$', 'number', x3)
    x5 = re.sub('[^\w\s]', '', x4)        
    words = x5
    words = words.split()
    words = [w for w in words if w in vocabulary]
    return words

new_file = open('pydictionarymodified3000mc.txt')
rawtext = new_file.read()
keywords = re.findall('(?<=\n).*(?=:\n)', rawtext)
words = []
for i in re.split('[\n].*:[\n]', rawtext):    #Generates a list of sentences tokenized and uncapitalized
    x = review_sentence(i, vocabulary)
    words.append(x)
words.pop(0)


def multiplesentences(sentences):
    """
    A function to split a list of sentences into a lists of lists of words in the sentences
    """
    definitions = []
    for i in sentences:
        definitionlist = []
        for j in i:
            if j != '':
                k = re.split(' ', j)
                k = [w for w in k if w != '']
                definitionlist.append(k)
        definitions.append(definitionlist)

#for i in rawwords:
#    k = [w for w in i if w]
#    definitions.append(k)
 
dictionary = list(zip(words, keywords))
dictionary.sort()
labeledsentences = []
dictwords = []
for i, j in dictionary:
    if i != []:
        labeledsentence = gensim.models.doc2vec.LabeledSentence(words=i, tags=j)
        if labeledsentence.tags not in dictwords:
            dictwords.append(labeledsentence.tags)
            labeledsentences.append(labeledsentence)

#def writedefinition(iterable, outfile):
#    appoutfile = open(outfile, 'a')
    

        


#At this point, some of the words in the original wikimodel vocab will no longer have definitions
import pickle as pk
vocab = []
for i in labeledsentences:
    vocab.append(i.tags)
with open('dictionaryvocab.pk', 'wb') as f:
    pk.dump(vocab, f)


#wikimodel.sg=0                    only if switching from dbow to dm
#wikimodel.cbow_words=0
wikimodel.cbow_mean=1
wikimodel.update_probabilities(labeledsentences)
wikimodel.docvecs.reset_weights(wikimodel)
wikimodel.regulardm = False                  #Leaves word vectors untrained
wikimodel.train(labeledsentences)