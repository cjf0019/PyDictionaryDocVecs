# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 13:26:05 2017
Allows the visualization of dictionary vectors, including their original
vocabulary vectors, concatenated with their corresponding docvecs. The resulting
concatenated vector receives dimensionality reduction via sklearn's tSNE. A control
model, a model of the same size as the concatenated vectors, without dictionary
training, is also included but commented out, the reason being, after training
on the Wikipedia corpus, I received a memory error when attempting tSNE 
dimension reduction. In the future, will look into adding a PCA option for
prior dimensional reduction. Also includes a 
"pickwords" function that allows one to type in words to compare. 
@author: InfiniteJest
"""
import doc2vecwordfixed
import numpy as np
import os
os.chdir('C:\\Users\\InfiniteJest\\Documents\\Python_Scripts')

model = doc2vecwordfixed.Doc2VecWordFixed.load('wiki100dmnolbls001samp3000mcdicdocvecs')
#controlmodel = doc2vecwordfixed.Doc2VecWordFixed.load('wiki200dbownolbls001samp')
#del controlmodel.docvecs

def concatenatedocandwordvectors(model, vector_size):
    combinedsyn0 = []
    combinedwordlist = []
    for word in list(model.vocab.keys()):
        if word in list(model.docvecs.doctags):
            docindex = list(model.docvecs.doctags).index(word)
            combinedsyn0.append(np.concatenate([model[word],model.docvecs[docindex]]))
    combinedsyn0 = np.vstack(combinedsyn0)
    for i in combinedsyn0:
        for j in range(len(model.syn0)):
            if np.array_equal(i[0:vector_size], model.syn0[j]):
                combinedwordlist.append(model.index2word[j])
    return combinedsyn0, combinedwordlist
    
finalvecs, wordlist = concatenatedocandwordvectors(model, 100)

from sklearn.manifold import TSNE
tsne = TSNE()
#Y = tsne.fit_transform(controlmodel.syn0)
Y2 = tsne.fit_transform(finalvecs)

def pickwords(model1, docvectransform, wordvectransform):
    words = set(model1.docvecs.doctags.keys())
    testinput = input("Type a list of words you would like to compare:")
    testwords = testinput.split(" ")
    for word in testwords:
        if word not in words:
            testwords.pop(testwords.index(word))
    testvocabvectors = []
    testdocvectors = []
    for word in testwords:
        testvocabvectors.append(model1[word])
        docindex = list(model1.docvecs.doctags).index(word)
        testdocvectors.append(model1.docvecs[docindex])
    
    modelindices = []
    modeldocindices = []
    for i in testwords:
        modelindices.append(list(model1.vocab).index(i))
        modeldocindices.append(list(model1.docvecs.doctags).index(i))
    mag = 3*len(testwords)
    import matplotlib.pyplot as plt    
    plt.scatter(mag*wordvectransform[modelindices[:], 0], mag*wordvectransform[modelindices[:], 1])
    for label, x, y in zip(testwords, mag*wordvectransform[modelindices[:], 0], mag*wordvectransform[modelindices[:], 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.show()

    plt.scatter(mag*docvectransform[modeldocindices[:], 0], mag*docvectransform[modeldocindices[:], 1])
    for label, x, y in zip(testwords, mag*docvectransform[modeldocindices[:], 0], mag*docvectransform[modeldocindices[:], 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.show()
    

#testwords = list(model.docvecs.doctags.keys())

testwords = ['king', 'queen', 'poor', 'rich', 'bother', 'exam', 'human', 'animal', 'tether', 'hint', 'fear', 'anxiety', 'lose', 'win', 'dog', 'cat', 'bird', 'mouse', 'big', 'large', 'boring', 'war', 'weapon', 'peace', 'prosperity']

import matplotlib.pyplot as plt 
mag = 5000
combinedmodelindices = []
controlindices = []
for word in testwords:
    if word in set(wordlist):
        combinedmodelindices.append(wordlist.index(word))
#    controlindices.append(list(controlmodel.vocab.keys()).index(word))
#plt.scatter(mag*Y[controlindices[:], 0], mag*Y[controlindices[:], 1])
#for label, x, y in zip(testwords, mag*Y[controlindices[:], 0], mag*Y[controlindices[:], 1]):
#    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
#plt.show()

plt.scatter(mag*Y2[combinedmodelindices[:], 0], mag*Y2[combinedmodelindices[:], 1])
for label, x, y in zip(testwords, mag*Y2[combinedmodelindices[:], 0], mag*Y2[combinedmodelindices[:], 1]):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
plt.show()