# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 20:10:32 2018

@author: InfiniteJest
"""

import sys
import re
import os
os.chdir('C:\\Users\\InfiniteJest\\Documents\\Python_Scripts')
from processparse import NLTKTreeNoPunct, xtrcttree
from NLTKRNTN import NLTKRNTN
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import numpy as np
import pickle
#### Load the word2vec model and word-index dictionaries
with open('wikimodelword2index.pickle', 'rb') as handle:
    word2index = pickle.load(handle)
with open('wikimodelindex2word.pickle', 'rb') as handle:
    index2word = pickle.load(handle)
wikimodel = np.load('wikimodelsyn0.npy')


definitionalwords = open('recnnwords5.txt')
treefile = open('recnndefinitions3.txt.OUT')
trees = []
a = 0
for word in definitionalwords:
    a += 1
    xtrcted = xtrcttree(treefile)
    tree = NLTKTreeNoPunct.fromstringnopunct(xtrcted)
    if word.strip('[\n]') in word2index.keys():
        for i in tree.subtrees():
    # Set the labels equal to the definitional words (is traditionally set to sentiment)
            i.set_label(word2index[word.strip('[\n]')])  #convert word to model index for label
#            i.set_label(word.strip('[\n]'))
        trees.append(tree)        




#### Test to generate embeddings from word2vec model
#x = tf.placeholder(tf.int32, shape=(None, 2))
#emb = tf.placeholder(tf.float32, shape=np.shape(wikimodel.syn0))
#embed = tf.nn.embedding_lookup(emb, x)
#init_op = tf.global_variables_initializer()
#feed_dict = {x: [[1, 2]], emb: wikimodel.syn0}
#session = tf.Session()
#session.run(embed, feed_dict)
########################################################


random.shuffle(trees)
train = trees[:42000]
test = trees[42000:]

V, D = np.shape(wikimodel)
K = D

model = NLTKRNTN(V, D, K, tf.nn.relu, word2vecmodel=wikimodel, word2index=word2index, \
                 index2word=index2word, fixembeddings=True)
model.fit(train)
print("train accuracy:", model.score(None))
print("test accuracy:", model.score(test))


#if __name__ == '__main__':
#    main()