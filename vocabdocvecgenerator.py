# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 21:31:46 2017

@author: InfiniteJest
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 16:08:24 2017

@author: InfiniteJest
"""

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
new_file = open('GutenbergDictionary3.txt')
rawtext = new_file.read()
rawkeywords = re.findall('(?<=\n).*[A-Z]{3,}', rawtext)
rawwords = []
for i in re.split('[\n][\n].*?[A-Z]{3,}.*[\n]', rawtext):    #Generates a list of sentences tokenized and uncapitalized
   x = i.lower()
   x = x.split()
   rawwords.append(x)

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
    
dictionarydict = dict(zip(dictionarywords, dictionarydefinitions))
dictvalues = list(dictionarydict.values())
dictkeys = list(dictionarydict.keys())
dictkeys.sort()

from PyDictionary import PyDictionary

class PyDictionaryMod(PyDictionary):
    def write_to_file(self):
        file = open('pydictionary.txt', 'a')
        try:
            dic = self.getMeanings()
            for key in dic.keys():
                file.write(key + ':')
                file.write("\n")
                try:
                    for k in dic[key].keys():
                        for m in dic[key][k]:   
                            file.write(m)
                            file.write("\n\n")
                except:
                    pass
        except:
            pass
        file.close()

#pydict = PyDictionaryMod(dictkeys)
#pydict.write_to_file()

#file = open('pydictionary.txt', 'r')
#x = file.read()
#x = re.sub('\(.*\n', '', x)
#x = re.sub('\n\n(?!.*\:\n)', '\n', x)
#x = re.sub('\;|\,', '', x)
#x = re.sub('(?=\n.*\:)', '\n', x)
#x = re.sub('\n.*\:\n\n', '', x)
#output = open('pydictionarymodified.txt', 'w')
#output.write(x)
#output.close()

new_file = open('pydictionarymodified.txt')
rawtext = new_file.read()
rawkeywords = re.findall('(?<=\n).*(?=:\n)', rawtext)
rawwords = []
for i in re.split('[\n][\n].*:[\n]', rawtext):    #Generates a list of sentences tokenized and uncapitalized
   x = i.lower()
   x = x.split('\n')
   rawwords.append(x)
   
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
    labeledsentences.append(gensim.models.doc2vec.LabeledSentence(words=i, tags=j))
    

def train_document_dm(model, doc_words, doctag_indexes, alpha, work=None, neu1=None,
                      learn_doctags=True, learn_words=True, learn_hidden=True,
                      word_vectors=None, word_locks=None, doctag_vectors=None, doctag_locks=None, dnn=False):
    """
                          Expanded to include documents as lists of lists. Also added extra-layer
                          deep neural network training (Set dnn = True). The new word vectors are comprised
                          of the network output up to, but not including the last hidden layer (kept as syn1,
                          just as in Doc2Vec)
    """

    print('This is the correct one! :)')
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
                    if dnn == True:                   
                        l1 /= count
                        neu1e = train_cbow_pair_dnn(model, word, word2_indexes, l1, alpha,
                                                                   learn_vectors=dnn, learn_hidden=learn_hidden)
                    else:
                        l1 /= count
                        neu1e = gensim.models.word2vec.train_cbow_pair(model, word, word2_indexes, l1, alpha,
                                                                   learn_vectors=False, learn_hidden=False)                        

                if not model.cbow_mean and count > 1:
                    neu1e /= count
                if (dnn and not learn_words) or learn_words:  #dnn will learn the word vectors anyways
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
                    if dnn == True:                   
                        l1 /= count
                        neu1e = train_cbow_pair_dnn(model, word, word2_indexes, l1, alpha,
                                                                   learn_vectors=dnn, learn_hidden=learn_hidden)
                    else:
                        l1 /= count
                        neu1e = gensim.models.word2vec.train_cbow_pair(model, word, word2_indexes, l1, alpha,
                                                                   learn_vectors=False, learn_hidden=False)                        

            if not model.cbow_mean and count > 1:
                neu1e /= count
            if (dnn and not learn_words) or learn_words:
                for i in word2_indexes:
                    word_vectors[i] += neu1e * word_locks[i]     
            if learn_doctags:
                for i in doctag_indexes:
                    doctag_vectors[i] += neu1e * doctag_locks[i]
    return len(word_vocabs)
    
    
def train_cbow_pair_dnn(model, word, input_word_indices, l1, alpha, learn_vectors=True, learn_hidden=True):
    neu1e = np.zeros(l1.shape)

    p1 = np.sum(model.syn2[input_word_indices], axis=0)           #the previous layer's saved input weights
    count = len(input_word_indices)
    p1 /= count
    if model.hs:
        p2a = model.syn3[word.point]                                #the previous layer's saved hidden weights
        fpa = 1. / (1. + np.exp(-np.dot(p1, p2a.T)))                #the previous layer's activation function
        l2a = model.syn1[word.point]  # 2d matrix, codelen x layer1_size
        fa = 1. / (1. + np.exp(-np.dot(fpa, l2a.T)))  # uses previous layer's output as input
        ga = (1. - word.code - fa) * alpha  # vector of error gradients multiplied by the learning rate
        if learn_hidden:
            model.syn1[word.point] += np.outer(ga, p1)  # learn hidden -> output
        neu1e += np.dot(ga, l2a)  # save error

    if model.negative:
        # use this word (label = 1) + `negative` other random words not from this sentence (label = 0)
        p2b = model.syn3neg[word.point]
        fpb = 1. / (1. + np.exp(-np.dot(p1, p2b.T)))                #the previous layer's activation function 
        word_indices = [word.index]
        while len(word_indices) < model.negative + 1:
            w = model.cum_table.searchsorted(model.random.randint(model.cum_table[-1]))
            if w != word.index:
                word_indices.append(w)
        l2b = model.syn1neg[word_indices]  # 2d matrix, k+1 x layer1_size
        fb = 1. / (1. + np.exp(-np.dot(fpb, l2b.T)))  # uses previous layer's output as input
        gb = (model.neg_labels - fb) * alpha  # vector of error gradients multiplied by the learning rate
        if learn_hidden:
            model.syn1neg[word_indices] += np.outer(gb, p1)  # learn hidden -> output
        neu1e += np.dot(gb, l2b)  # save error

    if learn_vectors:
        # learn input -> hidden, here for all words in the window separately
        if not model.cbow_mean and input_word_indices:
            neu1e /= len(input_word_indices)
        for i in input_word_indices:
            model.syn0[i] += neu1e * model.syn0_lockf[i]

    return neu1e

class Doc2VecWordFixed(gensim.models.doc2vec.Doc2Vec):
    """
    A modification of Doc2Vec that allows for extra-layer deep neural network training... in addition to
    the input, projection, hidden, and output layers, it allows for the addition of a second hidden layer.
    The first layer must be trained with dbow and the second with dm. The first layer, both projection and hidden,
    is saved as syn2 and syn3 respectively. The document probability distributions can then be updated and the
    vocabulary words fixed with 'update probabilities' for training on new documents. Note: the docvecs
    are only inferred, not trained by the DNN.
    """
    def __init__(self, documents=None, size=300, alpha=0.025, window=8, min_count=5,
                 max_vocab_size=None, sample=0, seed=1, workers=1, min_alpha=0.0001,
                 dm=1, hs=1, negative=0, dbow_words=0, dm_mean=0, dm_concat=0, dm_tag_count=1,
                 docvecs=None, docvecs_mapfile=None, comment=None, trim_rule=None, dnn=False, **kwargs):
        self.dnn = dnn
        super(Doc2VecWordFixed, self).__init__(documents=documents, size=size, alpha=alpha, window=window, min_count=min_count,
                max_vocab_size=max_vocab_size, sample=sample, seed=seed, workers=workers, min_alpha=min_alpha,
                dm=dm, hs=hs, negative=negative, dbow_words=dbow_words, dm_concat=dm_concat,
                dm_tag_count=dm_tag_count, docvecs=docvecs, docvecs_mapfile=docvecs_mapfile, comment=comment,
                trim_rule=trim_rule)
    
    def _do_train_job(self, job, alpha, inits):
        work, neu1 = inits
        dnn = self.dnn
        tally = 0
        for doc in job:
            indexed_doctags = self.docvecs.indexed_doctags(doc.tags)
            doctag_indexes, doctag_vectors, doctag_locks, ignored = indexed_doctags
            if dnn == True:
                ifdnn = True
            else:
                ifdnn = False
            if self.sg:
                tally += gensim.models.doc2vec.train_document_dbow(self, doc.words, doctag_indexes, alpha, work,
                                             train_words=self.dbow_words, doctag_vectors=doctag_vectors, doctag_locks=doctag_locks)
            elif self.dm_concat:
                tally += gensim.models.doc2vec.train_document_dm_concat(self, doc.words, doctag_indexes, alpha, work, neu1,
                                                  doctag_locks=doctag_locks, learn_words=ifdnn)
            else:
                tally += train_document_dm(self, doc.words, doctag_indexes, alpha, work, neu1, 
                                           doctag_locks=doctag_locks, learn_words=ifdnn, dnn=dnn)
            self.docvecs.trained_item(indexed_doctags)

        return tally, self._raw_word_count(job)

    def build_docvecs(self, documents):
        self.docvecs = gensim.models.doc2vec.DocvecsArray()
        document_no = -1
        for document_no, document in enumerate(documents):
            document_length = len(document.words)
#            for tag in document.tags:        NOTE: Previously EACH SENTENCE had its own tag in a document... modifying so only one tag per document.
            self.docvecs.note_doctag(document.tags, document_no, document_length)   #Changed tag to document.tags
            self.docvecs.reset_weights(self)

    def save_layer(self, documents): 
        savedlayer = self.syn0
        self.syn2 = savedlayer
        if hasattr(self, 'syn1'):
            savedhiddenlayer = self.syn1 
            self.syn3 = savedhiddenlayer
        if hasattr(self, 'syn1neg'):
            savedhiddenlayerneg = self.syn1neg
            self.syn3neg = savedhiddenlayerneg
        index2wordfirstlayer = self.index2word
        self.index2wordfirstlayer = index2wordfirstlayer

    def update_probabilities(self, documents, min_count=None, sample=None, dry_run=False, keep_raw_vocab=False, trim_rule=None):
        """
            Put together from pieces scale_vocab and finalize_vocab and then modified further,
            this will generate the output probability distributions for a new set of documents,
            without resetting the weights of the word or document vectors.
        """
        from six import iteritems
        from collections import defaultdict
        self.build_docvecs(labeledsentences)

        document_no = -1
        total_words = 0
        vocab = defaultdict(int)
        for document_no, document in enumerate(documents):
            document_length = len(document.words)
            if isinstance(document.words, list):          #added to include documents as lists of sentences
                for sentence in document.words:
                    for word in sentence:
                        vocab[word] += 1
                total_words += len(document.words)
            else:
                for word in document.words:
                    vocab[word] += 1
                total_words += len(document.words)               

        self.corpus_count = document_no + 1
        self.raw_vocab = vocab
        print(self.raw_vocab.keys())        

        min_count = min_count or self.min_count
        retain_total = 0
        sample = sample or self.sample
        if not dry_run:
            self.index2word = []
            # make stored settings match these applied settings
            self.min_count = min_count
            self.sample = sample
        #keep words found in BOTH previous layer's vocab and in current documents... necessary for accurate probabilities
        retain_words = list(filter(lambda x: x in list(self.vocab.keys()), self.raw_vocab.keys()))
        self.vocab = {}
        for word, v in iteritems(self.raw_vocab):
            if word in set(retain_words): 
                retain_total += v      #might want to reexamine v here
                if not dry_run:            
                    self.vocab[word] = gensim.models.word2vec.Vocab(count=v, index=len(self.index2word))
                    self.index2word.append(word)
            
        # Precalculate each vocabulary item's threshold for sampling
        if not sample:
            # no words downsampled
            threshold_count = retain_total
        elif sample < 1.0:
            # traditional meaning: set parameter as proportion of total
            threshold_count = sample * retain_total
        else:
            # new shorthand: sample >= 1 means downsample all words with higher count than sample
            threshold_count = int(sample * (3 + np.sqrt(5)) / 2)


        downsample_total, downsample_unique = 0, 0
        for w in retain_words:
            v = self.raw_vocab[w]
            word_probability = (np.sqrt(v / threshold_count) + 1) * (threshold_count / v)
            if word_probability < 1.0:
                downsample_unique += 1
                downsample_total += word_probability * v
            else:
                word_probability = 1.0
                downsample_total += v
            if not dry_run:
                self.vocab[w].sample_int = int(round(word_probability * 2**32))

        if not dry_run and not keep_raw_vocab:
            self.raw_vocab = defaultdict(int)        #delete raw vocab

        # return from each step: words-affected, resulting-corpus-size
        report_values = {'downsample_unique': downsample_unique, 'downsample_total': int(downsample_total)}
        # print extra memory estimates
        report_values['memory'] = self.estimate_memory(vocab_size=len(retain_words))
        if self.hs:
            # add info about each word's Huffman encoding
            self.create_binary_tree()
        if self.negative:
            # build the table for drawing random words (for negative sampling)
            self.make_cum_table()
        return report_values

    def setup_another_layer(self, documents):
        self.save_layer(documents)
        self.update_probabilities(documents)
        super(Doc2VecWordFixed, self).reset_weights()

    def fixed_reset_from(self, other_model):
        """
        Grabs vocab vectors from another model. Fixes the vectors, unlike the standard reset_from.
        """
        self.vocab = other_model.vocab
        self.index2word = other_model.index2word
        self.cum_table = other_model.cum_table
        
import wordvectorgenerator
model = Doc2VecWordFixed(wordvectorgenerator.labeledsentences2, size=100, workers=4, min_count=10, window=8, dm=0, dbow_words=1)
#del model.docvecs
#model.sg=0
#model.cbow_words=0
#model.cbow_mean=1
#model.update_probabilities(labeledsentences)
#model.train(labeledsentences)

combinedsyn0 = []
combinedwordlist = []
for word in list(model.vocab.keys()):
    if word in list(model.docvecs.doctags):
        docindex = list(model.docvecs.doctags).index(word)
        combinedsyn0.append(np.concatenate([model[word],model.docvecs[docindex]]))
combinedsyn0 = np.vstack(combinedsyn0)

for i in combinedsyn0:
    for j in range(len(model.syn0)):
        if np.array_equal(i[0:100], model.syn0[j]):
            combinedwordlist.append(model.index2word[j])

controlmodel = Doc2VecWordFixed(wordvectorgenerator.labeledsentences2, size=200, workers=4, min_count=10, window=8, dm=0, dbow_words=1)



from sklearn.manifold import TSNE
tsne = TSNE()
Y = tsne.fit_transform(controlmodel.syn0)
Y2 = tsne.fit_transform(combinedsyn0)

#testwords = list(model.docvecs.doctags.keys())

testwords = ['king', 'queen', 'poor', 'rich', 'bother', 'exam', 'human', 'animal', 'tether', 'hint', 'fear', 'anxiety', 'lose', 'win', 'dog', 'cat', 'bird', 'mouse', 'big', 'large', 'boring', 'war', 'weapon', 'peace', 'prosperity']

import matplotlib.pyplot as plt 
mag = 5000
combinedmodelindices = []
controlindices = []
for word in testwords:
    if word in set(combinedwordlist):
        combinedmodelindices.append(combinedwordlist.index(word))
        controlindices.append(list(controlmodel.vocab.keys()).index(word))
plt.scatter(mag*Y[controlindices[:], 0], mag*Y[controlindices[:], 1])
for label, x, y in zip(testwords, mag*Y[controlindices[:], 0], mag*Y[controlindices[:], 1]):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
plt.show()


plt.scatter(mag*Y2[combinedmodelindices[:], 0], mag*Y2[combinedmodelindices[:], 1])
for label, x, y in zip(testwords, mag*Y2[combinedmodelindices[:], 0], mag*Y2[combinedmodelindices[:], 1]):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
plt.show()

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