# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 20:10:32 2018

@author: InfiniteJest
"""

import numpy as np
import re
import os
from nltk.tree import Tree
os.chdir('C:\\Users\\InfiniteJest\\Documents\\Python_Scripts')
import gensim
from processparse import get_labels, NLTKRNTN, NLTKTreeNoPeriod
import doc2vecwordfixed

wikimodel = doc2vecwordfixed.Doc2VecWordFixed.load('wiki100dmnolbls001samp3000mc')
vocabfile = open('dictionaryvocabwords', 'r')
vocabfile = vocabfile.read()
vocabulary = vocabfile.split('\n')
vocabulary.sort()
vocabulary.pop(0)


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
    words = x5
    words = words.split()
    words = [w for w in words if w in vocabulary]
    words = ' '.join(words)+'.'
    return words

new_file = open('pydictionarymodified3000mc.txt')
rawtext = new_file.read()
recnndef = open('recnndefinitions.txt', 'a')
recnnwords = open('recnnwords.txt', 'a')
for i in re.split('[\n](?=.*:[\n])', rawtext):
    if i == '':
        pass
    else:
        vocabword = re.findall('.*(?=:\n)', i)
        j = re.sub('.*\:\n', '', i)
        k = [l+'.' for l in j.split('\n') if len(l) != 0]
        vocabword = vocabword*len(k)
        k = ' '.join(k)            
        recnndef.write(k+'\n')
        for i in vocabword:
            recnnwords.write(i+'\n')
recnndef.close()
recnnwords.close()


def xtrcttree(file):
    firstline = file.readline()
    print('firstline: ', firstline)
    numtokens = re.search('(?<=[\(])[0-9]+(?= token)', firstline)
    print(numtokens)
    numtokens = numtokens.group(0)
    for i in range(int(numtokens)+1):
        file.readline()
    j = 0
    tree = []
    while j != 1:
        treeline = file.readline()
        if re.search('root\(ROOT', treeline):
            j = 1
            depend = ''
            while not re.match('punct\(.*\, [^\:\`\'\-]', depend):
                depend = file.readline()
                print(depend)
            file.readline()
        else:
            tree.append(treeline)
    return ''.join(tree)



definitionalwords = open('recnnwords2.txt')
treefile = open('recnndefinitions1b.txt.out')
trees = []
for word in definitionalwords:
    xtrcted = xtrcttree(treefile)
    print('xtrcted', xtrcted)
    tree = NLTKTreeNoPeriod.fromstringnoperiod(xtrcted)
    for i in tree.subtrees():
        i.set_label(word.strip('[\n]'))
    trees.append(tree)

        

    
    
    