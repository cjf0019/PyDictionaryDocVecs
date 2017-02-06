# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 16:40:39 2016

Processes the Brown corpus and makes word and document vectors. 


@author: InfiniteJest
"""

import gensim as gs
import nltk
import re

x = nltk.corpus.brown.sents()

stops = set(nltk.corpus.stopwords.words("english"))
stops.update(['.',',','"',"'", '?', '!', ':', ';','(',')','[',']','{','}',"''",'``', 'c', 'o', 'pa', 'fromm', 'n', 'k.', 'sba', 'lou', 'gen.', "o'banion", 'h.', 'dec.', '**zg', 'figs.', '/', 'k', 'j.', 'i.', 'b.', 'h', 'du', 'v.', 'u.', 'r', 'p', 'x', 'u', 'm.', 'a.l.a.m.', 'o.', 'c.', 'r.', 'ml', 'g', 'p.', 'w.', 'n.', 'v', 'd.', '7th', 'gm.', 't.', 'g.', 'a.', 'milton'])


def review_to_wordlist( list_of_sentences, remove_stopwords=False ):
    a = []    
    for sentences in list_of_sentences:
        b = []
        for words in sentences:
            x = re.sub('\'s', '', words.lower())            
            x2 = re.sub('^\$\d*$', 'money', x)
            x3 = re.sub('^1\d\d\d$', 'year', x2)
            x4 = re.sub('^\d+$', 'number', x3)
            words = x4
            b.append(words)
        b = [w for w in b if not w in stops]
        a.append(b)
    return a

x = review_to_wordlist(x, remove_stopwords=True)   
print(x[1:100])

labeledsentences2 = []

for sentence in x:
    labeledsentences2.append(gs.models.doc2vec.LabeledSentence(words=sentence, tags=[x.index(sentence)]))


brownmodel= gs.models.doc2vec.Doc2Vec(labeledsentences2, dbow_words=1, size=100, window=8, min_count=10, workers=4)

#model = gs.models.Doc2Vec(x, size = 100, window = 5, min_count = 10, workers = 4, train_lbls = False)

#model = gs.models.word2vec.Word2Vec(x, size = 100, window = 5, min_count = 10, workers = 4)   
#gs.models.word2vec.BrownCorpus('C:\Users\InfiniteJest\Anaconda3\Lib\site-packages\nltk\corpus')