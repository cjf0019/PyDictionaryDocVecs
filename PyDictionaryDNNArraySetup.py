# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 14:45:50 2017
Sets up input and output arrays for a word vector deep neural network involving the PyDictionary 
definitions (originally found on WordNet). Word vectors are trained using a modified version of 
gensim's Doc2Vec (Doc2VecWordFixed). First, a vocabulary and vectors are generated using the Brown
Corpus and the Distributed Bag of Words Model. The docvecs are discarded and if the user
wishes replaced by the definitional words (by function 'build_docvecs'). 

The input of the neural network is the original vector for each word in the vocabulary
trained from a previous corpus (in my case wikipedia). The output is equal to the average of all of the vectors
found in each word's definition and can include the docvec (labeled by the defitional word...
not included here yet but would need to concatenate these vectors to the others).
The conditional probabilities of the words and their contexts are reread, this time 
from the PyDictionary text. Unlike the original Doc2Vec, the weights are not reset, thus fixing the 
vectors from the previous corpus run.

@author: InfiniteJest
"""


import numpy as np
#import BeautifulSoup
import re
import os
import nltk
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
stops = set(nltk.corpus.stopwords.words("english"))
stops.update(['.',',','"',"'", '?', '!', ':', ';','(',')','[',']','{','}',"''",'``'])
import doc2vecwordfixed
import gensim

os.chdir('C:\\Users\\InfiniteJest\\Documents\\Python_Scripts')

model = doc2vecwordfixed.Doc2VecWordFixed.load('wiki100dbownolbls001samp1000mc')
vocabulary = model.vocab.keys()


new_file = open('pydictionarymodified3000mc.txt')
rawtext = new_file.read()
rawkeywords = re.findall('(?<=\n).*(?=:\n)', rawtext)
rawwords = []
for i in re.split('[\n].*:[\n]', rawtext):    #Generates a list of sentences tokenized and uncapitalized
   x = review_sentence(i)
   x = x.split('\n')
   rawwords.append(x)
rawwords.pop(0)
  
definitions = []
for i in rawwords:
    definitionlist = []
    for j in i:
        if j != '':
            k = re.split(' ', j)
            k = [w for w in k if w != '']
            definitionlist.append(k)
    definitions.append(definitionlist)
    
dictionary = list(zip(definitions, rawkeywords))
dictionary.sort()
labeledsentences = []
for i, j in dictionary:
    if i != []:
        labeledsentences.append(gensim.models.doc2vec.LabeledSentence(words=i, tags=j))

fullsentencelist= []
fulltaglist = []
for definition in labeledsentences:
    for sentence in definition.words:
        fullsentencelist.append(sentence)
    count = 0
    while count < len(definition.words):
        fulltaglist.append(definition.tags)
        count += 1


model.update_probabilities(labeledsentences)

word_vectors = model.syn0
out = np.array([])
words = []
passedcount = 0
for sentence in labeledsentences[0].words:    
    word_vocabs = [model.vocab[w] for w in sentence if w in model.vocab and
    model.vocab[w].sample_int > model.random.rand() * 2**32]
    for pos, word in enumerate(word_vocabs):
        reduced_window = model.random.randint(model.window) 
        start = max(0, pos - model.window + reduced_window)
        window_pos = enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start)
        word2_indexes = [word2.index for pos2, word2 in window_pos if pos2 != pos]
        l1 = np.sum(word_vectors[word2_indexes], axis=0)
        count = len(word2_indexes)
        if labeledsentences[0].words.index(sentence) == 0:
            out = np.append(out, l1/count)
        else:
            out = np.vstack([out,l1/count])
        words.append()
for definition in labeledsentences:
    if definition.tags not in set(model.vocab.keys()):
        pass
    else:
        for sentence in definition.words:
            word_vocabs = [model.vocab[w] for w in sentence if w in model.vocab and
            model.vocab[w].sample_int > model.random.rand() * 2**32]
            for pos, word in enumerate(word_vocabs):
                reduced_window = model.random.randint(model.window) 
                start = max(0, pos - model.window + reduced_window)
                window_pos = enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start)
                word2_indexes = [word2.index for pos2, word2 in window_pos if pos2 != pos]
                l1 = np.sum(word_vectors[word2_indexes], axis=0)
                count = len(word2_indexes)
                if np.isnan(l1/count).any():
                    passedcount += 1
                elif len(out) == 0:
                    out = np.append(out, l1/count)
                    words.append(definition.tags)
                else:
                    out = np.vstack([out, l1/count])
                    words.append(definition.tags)

print('finished out')

inp = np.array([])
for word in words:
    x = list(model.vocab).index(word)    
    if len(inp) == 0:
        inp = np.append(inp, model.syn0[x])
    else:    
        inp = np.vstack([inp, model.syn0[x]])

print('finished inp')

np.save('pydicdnninpwiki100.npy', inp)
np.save('pydicdnnoutwiki100.npy', out)
