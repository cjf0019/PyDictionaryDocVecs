# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 15:22:46 2018

@author: InfiniteJest
"""

import re
import os
from nltk.tree import Tree
os.chdir('C:\\Users\\InfiniteJest\\Documents\\Python_Scripts')
import gensim
from processparse import get_labels, NLTKRNTN, NLTKTreeNoPunct
import doc2vecwordfixed
import pickle

wikimodel = doc2vecwordfixed.Doc2VecWordFixed.load('wiki100dmnolbls001samp3000mc')
vocabfile = open('dictionaryvocabwords', 'r')
vocabfile = vocabfile.read()
vocabulary = vocabfile.split('\n')
vocabulary.sort()
vocabulary.pop(0)
vocabulary = [i for i in vocabulary if i in list(wikimodel.vocab.keys())]


##############################################################################
######The following is for storing dictionary of word2index and index2word 
######for easy access during training and to avoid having to import a full
###### Gensim word2vec model. Also saves the word2vec syn0 numpy array.
def wordorindex(model, wordorindex = 'index'):
    indices = []
    vocab = list(model.vocab.keys())
    for word in vocab:
        indices.append(vocab.index(word))
    if wordorindex == 'index':
        word2index = dict(zip(vocab, indices))
        return word2index
    elif wordorindex == 'word':
        index2word = dict(zip(indices, vocab))
        return index2word

word2index = wordorindex(wikimodel)
index2word = wordorindex(wikimodel, wordorindex = 'word')

with open('wikimodelword2index.pickle', 'wb') as handle:
    pickle.dump(word2index, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('wikimodelindex2word.pickle', 'wb') as handle:
    pickle.dump(index2word, handle, protocol=pickle.HIGHEST_PROTOCOL)
np.save('wikimodelsyn0.npy', wikimodel.syn0)

#############################################################################
#### The following was used to remove duplicate entries in the text file 
duplicates = []
vocabulary2 = []
for i in vocabulary:
    if i in vocabulary2:
        duplicates.append(i)
    else:
        vocabulary2.append(i)

definitionsfile = open('pydictionarymodified3000mcjan24.txt')
def removetextblock(openfile, writefile, beginreg, endreg):
    remove = False
    for line in openfile:
        if remove != True:
            writefile.write(line)
            remove = re.search(beginreg, line) != None
        elif remove == True:
            remove = (re.search(endreg, line) != None) == False
            pass

def removeduplicatedictentry(openfile, writefile, endreg, duplicatelist):
    remove = False
    duplicatereg = ['('+word+':)' for word in duplicatelist]
    duplicatereg = '|'.join(duplicatereg)
    beginreg = ''   #start as empty to allow one of each duplicated in the final file
    for line in openfile:
        if remove != True:
            writefile.write(line)
            beginsearch = re.search(beginreg, line)
            #append beginreg to indicate a future duplicate has been written once already
            if (beginsearch is None) or (beginsearch.group() == ''):
                firstdup = re.search(duplicatereg, line)
                if firstdup is not None:
                    beginreg = beginreg + '|'+str(firstdup.group())
            remove = (beginsearch != None) and (beginsearch.group() != '') #set to true if a duplicate word match found
        elif remove == True:
            endsearch = re.search(endreg, line)
            dupsearch = re.search(beginreg, line)
            remove = ((endsearch != None) and (dupsearch != None)) == False
            if remove == False:
                writefile.write(line)
    return

endreg = '[a-zA-Z]+\:[\n]'
newdeffile = open('pydictionarymodified3000mcjan22dupemoved.txt', 'a')
removeduplicatedictentry(definitionsfile, newdeffile, endreg, duplicates)
newdeffile.close()

endreg = '(?=[\n][a-zA-Z]+\:[\n])'
for duplicate in duplicates[0:2]:
    duplicatereg = duplicate +'\:[\n]'+'.+'+endreg
    hit = re.search(duplicatereg, rawtext, flags=re.DOTALL).group(0)
    rawtext = re.sub(hit, '', rawtext, count=1)
    
###Split By Definitions to Remove Exact Duplicates
rawlist = re.split('[\n](?=[a-zA-Z]+\:[\n])', rawtext, flags=re.DOTALL)
duplicates = []
rawlist2 = []
for definition in rawlist:
    if definition not in rawlist2:
        rawlist2.append(definition)
    else:
        duplicates.append(definition)

###Remove All Separate Duplicate Entries of the same word, which gets ones with switched around definitions
rawlist = re.split('[\n](?=[a-zA-Z]+\:[\n])', rawtext, flags=re.DOTALL)
duplicates = []
rawlist2 = []
words = []
for definition in rawlist:
    word = re.split('\:\n', definition)[0]
    if word not in rawlist2:
        rawlist2.append(definition)
        words.append(word)
    else:
        duplicates.append(word)
        
###############################################################################




    
   
def removetextblock(openfile, writefile, beginreg, endreg):
    beginremove = False
    while removed != True:
        line = openfile.readline()
        removed = re.search(beginremove, line)
        if removed != True:
            writefile.write(line)     


def review_sentence(sentence, vocabulary):
    """
    SPECIAL NOTE:
    Could mess with x5 if wanting to keep each definition corresponding to the same word separate
    """
    x = sentence.lower()
    x2 = re.sub('^\$\d*$', 'money', x)
    x3 = re.sub('^1\d\d\d$', 'year', x2)
    x4 = re.sub('^\d+$', 'number', x3)
    x5 = re.sub('[^\w\s]', '', x4)        
    x6 = re.sub("\'s", '', x5)
    x7 = re.sub("\'|\`", '', x6)
#    x8 = re.sub("(?<=[\s])[a-zA-Z]{1}(?=[\n])", "", x7)
    words = x7
    words = words.split()
    words = [w for w in words if w in vocabulary]
    words = ' '.join(words)+'.'
    return words

new_file = open('pydictionarymodified3000mcjan25.txt')
rawtext = new_file.read()
#rawtext = review_sentence(rawtext, vocabulary)
recnndef = open('recnndefinitions3.txt', 'a')
recnnwords = open('recnnwords4.txt', 'a')
numwords = 0
numdefinitions = 0
for i in re.split('[\n](?=.*:[\n])', rawtext):
    if i == '':
        pass
    else:
        vocabword = re.findall('.*(?=:\n)', i)
        j = re.sub('.*\:\n', '', i)
        k = [review_sentence(l, vocabulary) for l in j.split('\n')]  
        k = [l for l in k if len(l) > 1]   #keep only definitions with at least two words
        if len(k) >= 1:   #remove now empty entries
            vocabword = vocabword*len(k)
            numwords += len(vocabword)
            numdefinitions += len(k)
            k = ' '.join(k)            
            recnndef.write(k+'\n')
            for i in vocabword:
                recnnwords.write(i+'\n')
recnndef.close()
recnnwords.close()


def getmodelindices(model, wordlist):
    indices = []
    for word in wordlist:
        indices.append(list(model.vocab.keys()).index(word))
    return indices


def xtrctword2vecarray(model, vocabulary):
    print('The following words were in the vocabulary but not in the model:')
    notinmodel = [i for i in vocabulary if i not in list(model.vocab.keys())]
    print(notinmodel)
    print(len(notinmodel))
    vocabulary = [i for i in vocabulary if i in list(model.vocab.keys())]   #remove non-model words
    word_vectors = model.syn0
    word2index = {}
    vocabarray = []
    for pos, word in enumerate(vocabulary):
            vocabarray.append(model[word])
            word2index[word] = pos
    vocabarray = np.vstack(vocabarray)
    return vocabarray, word2index
            




recnndef = open('recnndefinitions.txt')
recnndef.close()
z = open('recnnwords.txt')
z.readline()
test = re.findall('.*(?=:\n)', rawtext)
test = [i for i in test if i != '']
len(test)
test2 = re.split('[\n](?=.*:[\n])', rawtext)
test2 = re.split('[\n](?=.*:[\n])', rawtext)
len(test2)
