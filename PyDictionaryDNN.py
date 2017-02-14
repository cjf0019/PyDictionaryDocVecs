# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 10:35:28 2017

Takes the input and output arrays generated from PyDictionaryDNNArraySetup.py and 
trains a neural network using keras. Here, it has one hidden layer with a sigmoid 
activation function and one hundred neurons. The input is the definitional word's 
vector (its Doc2Vec syn0 array), and the output is the average of the vectors of the
words found in the definition. Docvecs for each definitional word as part of the
 output are also possible but not yet added. 20% of the (shuffled) definitions 
 were used for validation.

@author: InfiniteJest
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import numpy as np
import os
os.chdir('C:\\Users\\InfiniteJest\\Documents\\Python_Scripts')

inp = np.load('pydicdnninp.npy')
out = np.load('pydicdnnout.npy')

dnn = Sequential()
#early_stopping = EarlyStopping(monitor='val_loss', patience=3)
dnn.add(Dense(100, input_dim=100, init='uniform', activation='sigmoid'))
dnn.add(Dense(100))
dnn.compile(loss='binary_crossentropy', optimizer='adam')
dnn.fit(inp, out, shuffle=True, validation_split=0.2, nb_epoch=100, batch_size=1)
