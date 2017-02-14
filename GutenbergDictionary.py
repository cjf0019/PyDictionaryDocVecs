# -*- coding: utf-8 -*-
"""
Spyder Editor


"""

#import pandas as pd
import numpy as np
#import BeautifulSoup
#import nltk
import re
import os
import multiprocessing
import nltk
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
stops = set(nltk.corpus.stopwords.words("english"))
stops.update(['.',',','"',"'", '?', '!', ':', ';','(',')','[',']','{','}',"''",'``'])

import gensim

os.chdir('C:\\Users\\InfiniteJest\\Documents\\Python_Scripts')
#file = open('GutenbergDictionary.txt')
#rawtext = file.read()

#fp = open('GutenbergDictionary2.txt')    
#x = fp.read()
#x = re.sub('(?<=[^\n])[\n](?=[^\n])',' ', x)
#x = re.sub('(?<=[A-Z]{3} ).*(?=[\n][\n]Def|[\n][\n]1\.)*', '', x)
#x = re.sub('[\n]Note\:.*', '', x)
#x = re.sub("[\n]\d\.|[\n]Defn\:", "", x)
#x = re.sub('\-\-.*', '', x)
#x = re.sub('\[Obs\.\].*?(?=[\n])', '', x)
#x = re.sub('Syn\.', '', x)
#x = re.sub('\[.*?\]', '', x)
#x = re.sub("\(.*?\)", "", x)
#x = re.sub('See', "", x)
#x = re.sub('Sir W. Hamilton.', "", x)
#x = re.sub('Milton.', "", x)
#x = re.sub('Fleming.', "", x)
#x = re.sub('Dryden.', "", x)
#x = re.sub('Shak.', "", x)
#x = re.sub('Chaucer', "", x)
#x = re.sub('(?<=\n).*?[A-Z]{3,}.*[\n][\n]', "", x)
#x = re.sub('Johnson.', "", x)
#x = re.sub('(?<=\n).*?[A-Z]{3,}.*[\n](?=[.*?[A-Z]{3,})', "", x)
#x = re.sub('(?<=\n).*[A-Z]{3,}.*;.*(?=\n)', "", x)
#x = re.sub('III', "", x)
#x = re.sub('XX', "", x)
#x = re.sub('XV', "", x)
#x = re.sub('VII', "", x)
#x = re.sub('(?<=[A-Z]{3}\n).*[a-z]*.*(?=\n[A-Z]{3})', '(?<=[A-Z]{3}\n).*[a-z]*.*\n(?=\n[A-Z]{3})', x)
#fp.close()
#new_file = open('GutenbergDictionary3.txt', 'w')
#new_file.write(x)
#new_file.close()

new_file = open('GutenbergDictionary3.txt')
rawtext = new_file.read()
rawkeywords = re.findall('(?<=\n).*[A-Z]{3,}', rawtext)
rawwords = []
for i in re.split('[\n][\n].*?[A-Z]{3,}.*[\n]', rawtext):    #Generates a list of sentences tokenized and uncapitalized
   x = i.lower()
   x = x.split()
   rawwords.append(x)
#print(rawwords)

def nestedlistfilter(nestedlist, dostops = False):      #Input various regular expressions to substitute out of a list.
    sub = str(input("Input a regular expression to filter out. If nothing, type 'Done': " ))
    while sub != 'Done':
        a = []    
        for sentence in nestedlist:
            if dostops == True:
                b = [w for w in sentence if w not in stops and w != sub]
            else:
                b = [w for w in sentence if w != sub]
            a.append(b)
        sub = str(input("Input a regular expression to filter out. If nothing, type 'Done': "))
        nestedlist = a
    return nestedlist

brownmodel = gensim.models.doc2vec.Doc2Vec.load('brownindocvecs')
vocabulary = brownmodel.vocab.keys()


def review_to_wordlist( list_of_sentences ):
    a = []    
    for sentences in list_of_sentences:
        b = []
        for words in sentences:
            x = re.sub("\'s", '', words.lower())           
            x2 = re.sub('^\$\d*$', 'money', x)
            x3 = re.sub('^1\d\d\d$', 'year', x2)
            x4 = re.sub('^\d+$', 'number', x3)
            x5 = re.sub('[^\w\s]', '', x4)
            words = x5
            b.append(words)
        b = [w for w in b if w in vocabulary]
        a.append(b)
    return a


def train_document_dm(model, doc_words, doctag_indexes, alpha, work=None, neu1=None,
                      learn_doctags=True, learn_words=True, learn_hidden=True,
                      word_vectors=None, word_locks=None, doctag_vectors=None, doctag_locks=None):
    from numpy import dot    #Added so it could properly access word2vec
    if word_vectors is None:
        word_vectors = model.syn0
    if word_locks is None:
        word_locks = model.syn0_lockf
    if doctag_vectors is None:
        doctag_vectors = model.docvecs.doctag_syn0
    if doctag_locks is None:
        doctag_locks = model.docvecs.doctag_syn0_lockf

    if isinstance(doc_words[0], list):
        for sentence in doc_words:
            word_vocabs = [model.vocab[w] for w in sentence if w in model.vocab and
            model.vocab[w].sample_int > model.random.rand() * 2**32]
            for pos, word in enumerate(word_vocabs):
                reduced_window = model.random.randint(model.window)  # `b` in the original doc2vec code
                start = max(0, pos - model.window + reduced_window)
                window_pos = enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start)
                word2_indexes = [word2.index for pos2, word2 in window_pos if pos2 != pos]
                l1 = np.sum(word_vectors[word2_indexes], axis=0) + np.sum(doctag_vectors[doctag_indexes], axis=0)
                count = len(word2_indexes) + len(doctag_indexes)
                if model.cbow_mean and count > 1 :
                    l1 /= count
                    neu1e = gensim.models.word2vec.train_cbow_pair(model, word, word2_indexes, l1, alpha,
                                                                   learn_vectors=False, learn_hidden=learn_hidden)
                if not model.cbow_mean and count > 1:
                    neu1e /= count
                if learn_words:
                    for i in word2_indexes:
                        word_vectors[i] += neu1e * word_locks[i]     
                if learn_doctags:
                    for i in doctag_indexes:
                        doctag_vectors[i] += neu1e * doctag_locks[i]
    else:
        word_vocabs = [model.vocab[w] for w in doc_words if w in model.vocab and
        model.vocab[w].sample_int > model.random.rand() * 2**32]
        for pos, word in enumerate(word_vocabs):
            reduced_window = model.random.randint(model.window)  # `b` in the original doc2vec code
            start = max(0, pos - model.window + reduced_window)
            window_pos = enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start)
            word2_indexes = [word2.index for pos2, word2 in window_pos if pos2 != pos]
            l1 = np.sum(word_vectors[word2_indexes], axis=0) + np.sum(doctag_vectors[doctag_indexes], axis=0)
            count = len(word2_indexes) + len(doctag_indexes)
            if model.cbow_mean and count > 1 :
                l1 /= count
                neu1e = gensim.models.word2vec.train_cbow_pair(model, word, word2_indexes, l1, alpha,
                                        learn_vectors=False, learn_hidden=learn_hidden)
            if not model.cbow_mean and count > 1:
                neu1e /= count
            if learn_words:
                for i in word2_indexes:
                    word_vectors[i] += neu1e * word_locks[i]     
            if learn_doctags:
                for i in doctag_indexes:
                    doctag_vectors[i] += neu1e * doctag_locks[i]
    return len(word_vocabs)



definitions = review_to_wordlist(rawwords)

keywords = []
for word in rawkeywords:
    keywords.append(word.split('; '))

words = review_to_wordlist(keywords)

dictionary = list(zip(words, definitions))
dictionary = [l for l in dictionary if l[0] != []]
dictionary = [l for l in dictionary if l[1] != []]
dictionarywords = []
dictionarydefinitions = []
for i in range(len(dictionary)):
    dictionarywords.append(dictionary[i][0][0])
    dictionarydefinitions.append(dictionary[i][1])

print(definitions[1:2])
print(words[1:2])
print('The amount words is:', len(words))
print('The amount of definitions is:', len(definitions))

dictionarydict = dict(zip(dictionarywords, dictionarydefinitions))
dictvalues = list(dictionarydict.values())
dictkeys = list(dictionarydict.keys())
dictkeys.sort()

dictionarydictlist = list(zip(dictionarywords, dictionarydefinitions))
 
combineddef = []
for i in range(len(dictkeys)):
     combineddef.append([])

    
for word, definition in zip(dictionarywords, dictionarydefinitions):
    index = dictkeys.index(word)
    combineddef[index].append(definition)

labeledsentences = []
finalized = zip(combineddef, dictkeys)
for i, j in finalized:
    labeledsentences.append(gensim.models.doc2vec.LabeledSentence(words=i, tags=j))

class Doc2VecDocFixed(gensim.models.doc2vec.Doc2Vec):
    def _do_train_job(self, job, alpha, inits):
        work, neu1 = inits
        tally = 0
        for doc in job:
            indexed_doctags = self.docvecs.indexed_doctags(doc.tags)
            doctag_indexes, doctag_vectors, doctag_locks, ignored = indexed_doctags
            if self.sg:
                tally += train_document_dbow(self, doc.words, doctag_indexes, alpha, work,
                                             train_words=self.dbow_words,
                                             doctag_vectors=doctag_vectors, doctag_locks=doctag_locks, learn_doctags=False)
            elif self.dm_concat:
                tally += train_document_dm_concat(self, doc.words, doctag_indexes, alpha, work, neu1,
                                                  doctag_vectors=doctag_vectors, doctag_locks=doctag_locks, learn_doctags=False)
            else:
                tally += train_document_dm(self, doc.words, doctag_indexes, alpha, work, neu1,
                                           doctag_vectors=doctag_vectors, doctag_locks=doctag_locks, learn_doctags=False)
            self.docvecs.trained_item(indexed_doctags)
        return tally, self._raw_word_count(job)

class Doc2VecWordFixed(gensim.models.doc2vec.Doc2Vec):
#    vocablist = list(self.vocab.keys())    
    def _do_train_job(self, job, alpha, inits):
        work, neu1 = inits
        tally = 0
        for doc in job:
            indexed_doctags = self.docvecs.indexed_doctags(doc.tags)
            doctag_indexes, doctag_vectors, doctag_locks, ignored = indexed_doctags
            if self.sg:
                tally += gensim.models.doc2vec.train_document_dbow(self, doc.words, doctag_indexes, alpha, work,
                                             train_words=self.dbow_words, doctag_vectors=doctag_vectors, doctag_locks=doctag_locks)
            elif self.dm_concat:
                tally += gensim.models.doc2vec.train_document_dm_concat(self, doc.words, doctag_indexes, alpha, work, neu1,
                                                  doctag_locks=doctag_locks, learn_words=False)
            else:
                tally += train_document_dm(self, doc.words, doctag_indexes, alpha, work, neu1, 
                                           doctag_locks=doctag_locks, learn_words=False)
            self.docvecs.trained_item(indexed_doctags)
#            self.syn0[vocablist.index(doc[1])] = self.docvecs[doc[1]]
        return tally, self._raw_word_count(job)

    def build_docvecs(self, documents):
        document_no = -1
        for document_no, document in enumerate(documents):
            document_length = len(document.words)
#            for tag in document.tags:        NOTE: Previously EACH SENTENCE had its own tag in a document... modifying so only one tag per document.
            self.docvecs.note_doctag(document.tags, document_no, document_length)   #Changed tag to document.tags
            self.docvecs.reset_weights(self)

    def fixed_reset_from(self, other_model):
        """
        Grabs vocab vectors from another model. Fixes the vectors, unlike the standard reset_from.
        """
        self.vocab = other_model.vocab
        self.index2word = other_model.index2word
        self.cum_table = other_model.cum_table
 
#import wordvectorgenerator
#model = Doc2VecWordFixed(wordvectorgenerator.labeledsentences2, size=100, workers=4, min_count=10, window=8, dm=0, dbow_words=1)
#del model.docvecs
#model.docvecs = gensim.models.doc2vec.DocvecsArray()
#model.sg=0
#model.cbow_words=0
#model.cbow_mean=1
#model.build_docvecs(labeledsentences)
#model.train(labeledsentences)

from sklearn.manifold import TSNE
tsne = TSNE()
#Y = tsne.fit_transform(model.syn0)
#Z = tsne.fit_transform(model.docvecs.doctag_syn0)

import matplotlib.pyplot as plt
plt.scatter(Y[:, 0], Y[:, 1])
for label, x, y in zip(model.index2word, Y[:, 0], Y[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
plt.show()


#finalwords = []
#finaldefinitions = []
#for entry in range(len(dictionary)):
#    empty = []
#    finaldefinitions.append(empty)
#for entry in range(len(dictionary)):    
#    a = range(len(dictionary[entry][1]))     
#    if dictionary[entry][0][0] in finalwords:
#        print(entry)        
#        definitions = []        
#        x = finalwords.index(dictionary[entry][0][0])        
#        print(x)        
#        for word in a:
#            definitions.append(dictionary[entry][1][word])
#        finaldefinitions[x].append(definitions)
#    b = range(len(dictionary[entry][1]))
#    if dictionary[entry][0][0] not in finalwords:
#        print(entry, 'not')        
#        definitions = []        
#        finalwords.append(dictionary[entry][0][0])
#        for word in b:        
#            definitions.append(dictionary[entry][1][word])
#        finaldefinitions[entry].append(definitions)

#finaldefinitions = [entry for entry in finaldefinitions if entry != []]
#print(finalwords)
#print(finaldefinitions)


def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    #review_text = re.sub(")    
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))   

   
def file_block(fp, number_of_blocks, block):
    '''
    Splits a file into blocks and iterates
    over the lines of one of the blocks.
 
    '''
 
    assert 0 <= block and block < number_of_blocks
    assert 0 < number_of_blocks
 
    fp.seek(0,2)
    file_size = fp.tell()
 
    ini = file_size * block / number_of_blocks
    end = file_size * (1 + block) / number_of_blocks
 
    if ini <= 0:
        fp.seek(0)
    else:
        fp.seek(ini-1)
        fp.readline()
 
    while fp.tell() < end:
        return fp.read()

def worker1(chunk_number):
    number_of_chunks = 4    
    fp = open('DictionaryTest.txt')    
    x = file_block(fp, number_of_chunks, chunk_number)
    x = re.sub('(?<=[^\n])[\n](?=[^\n])',' ', x)
    x = re.sub('(?<=[A-Z]{3} ).*(?=[\n][\n]Def|[\n][\n]1\.)*', '', x)    
    x = re.sub('[\n]Note\:.*', '', x)
    x = re.sub("[\n]\d\.|[\n]Defn\:", "", x)
    x = re.sub('\-\-.*', '', x)    
    x = re.sub('\[.*?\]', '', x)
    x = re.sub("\(.*?\)", "", x)
    fp.close()
    return x
    
def worker2(chunk_number):
    number_of_chunks = 4
    fp = open('DictionaryTest1.txt')
    x = file_block(fp, number_of_chunks, chunk_number)
    return re.sub('(?<=[A-Z]{3}[\n]).*(?=[\n]Def|1\.)', '', x)
    
def worker3(chunk_number):
    number_of_chunks = 4
    fp = open('DictionaryTest2.txt')
    x = file_block(fp, number_of_chunks, chunk_number)
    return re.sub('[\n]Note\:.*', '', x)
    
def worker4(chunk_number):
    number_of_chunks = 4
    fp = open('DictionaryTest3.txt')
    x = file_block(fp, number_of_chunks, chunk_number)
    return re.sub("[\n]\d\.|[\n]Defn\:", "", x)
    
def worker5(chunk_number):
    number_of_chunks = 4
    fp = open('DictionaryTest4.txt')
    x = file_block(fp, number_of_chunks, chunk_number)
    return re.sub('\-\-.*', '', x)

def worker6(chunk_number):
    number_of_chunks = 4
    fp = open('DictionaryTest5.txt')
    x = file_block(fp, number_of_chunks, chunk_number)
    return re.sub('\[.*?\]', '', x)

def worker7(chunk_number):
    number_of_chunks = 4
    fp = open('DictionaryTest6.txt')
    x = file_block(fp, number_of_chunks, chunk_number)
    return re.sub("\(.*?\)", "", x)

if __name__ == '__main__':
    pool = multiprocessing.Pool(4)      
#    if os.path.isfile('DictionaryTest1.txt') == False:    
#        x = pool.map(worker1, [0, 1, 2, 3])
#        for chunk in x:
#            new_file = open('DictionaryTest1.txt', "a+")
#            new_file.write(chunk)
#            new_file.close()
    if os.path.isfile('DictionaryTest1.txt') == False:
        x = pool.map(worker1, [0, 1, 2, 3])
        for chunk in x:
            new_file = open('DictionaryTest1.txt', 'a+')
            new_file.write(chunk)
            new_file.close()
#    if os.path.isfile('DictionaryTest3.txt') == False:
#        x = pool.map(worker3, [0, 1, 2, 3])
#        for chunk in x:
#            new_file = open('DictionaryTest3.txt', 'a+')
#            new_file.write(chunk)
#            new_file.close()
#    if os.path.isfile('DictionaryTest4.txt') == False:
#        x = pool.map(worker4, [0, 1, 2, 3])
#        for chunk in x:
#            new_file = open('DictionaryTest4.txt', 'a+')
#            new_file.write(chunk)
#            new_file.close()
#    if os.path.isfile('DictionaryTest5.txt') == False:
#        x = pool.map(worker5, [0, 1, 2, 3])
#        for chunk in x:
#            new_file = open('DictionaryTest5.txt', 'a+')
#            new_file.write(chunk)
#            new_file.close()
#    if os.path.isfile('DictionaryTest6.txt') == False:
#        x = pool.map(worker6, [0, 1, 2, 3])
#        for chunk in x:
#            new_file = open('DictionaryTest6.txt', 'a+')
#            new_file.write(chunk)
#            new_file.close()
#    if os.path.isfile('DictionaryTest7.txt') == False:
#        x = pool.map(worker7, [0, 1, 2, 3])
#        for chunk in x:
#            new_file = open('DictionaryTest7.txt', 'a+')
#            new_file.write(chunk)
#            new_file.close()
#    if os.path.isfile('DictionaryTest8.txt') == False:
#        x = pool.map(worker8, [0, 1, 2, 3])
#        for chunk in x:
#            new_file = open('DictionaryTest8.txt', 'a+')
#            new_file.write(chunk)
#            new_file.close()