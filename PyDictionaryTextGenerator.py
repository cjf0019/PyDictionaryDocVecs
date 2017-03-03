# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 10:11:38 2017
Outputs a text file with PyDictionary definitions, which are queried online on WordNet. The definitions queried either
come from a minimum count from a text file, or they come from a previously trained gensim Doc2Vec model. To run, input 
the script name followed by the directory to output the files, the input file name (either a text file or Doc2Vec model),
the min_count a word must meet from the corpus, and the output file name as the arguments. Will produce two files, one of
which will be designated "modified" to ensure the resulting dictionary is in the format

'word1:
definition1

word2:
definition2'

etc. NOTE: The file must be preprocessed so that each sentence is on a new line 

@author: InfiniteJest
"""

import re
import os
import nltk
from sys import argv
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
stops = set(nltk.corpus.stopwords.words("english"))
stops.update(['.',',','"',"'", '?', '!', ':', ';','(',')','[',']','{','}',"''",'``'])
import doc2vecwordfixed
import gensim

script, directory, inp, num, out = argv
os.chdir(str(directory))
if re.match('.*\.txt', str(inp)):
    model = gensim.models.doc2vec.Doc2Vec()
    sentences = gensim.models.doc2vec.TaggedLineDocument(str(inp))
    model.scan_vocab(inp)
    model.scale_vocab(num)
    vocabulary = list(model.vocab.keys())

else:
    file = doc2vecwordfixed.Doc2VecWordFixed.load(str(inp))
    vocabulary = list(file.vocab.keys())


from PyDictionary import PyDictionary
class PyDictionaryMod(PyDictionary):          #Added write_to_file function
    def write_to_file(self, output):
        file = open(str(output), 'a')
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

pydict = PyDictionaryMod(vocabulary)
pydict.write_to_file(out)

file = open(str(out), 'r')
x = file.read()
x = re.sub('\(.*\n', '', x)
x = re.sub('\n\n(?!.*\:\n)', '\n', x)
x = re.sub('\;|\,', '', x)
x = re.sub('(?=\n.*\:)', '\n', x)
x = re.sub('\n.*\:\n\n', '', x)
x = re.sub('\n.*\:(?=\n.*\:\n)', '', x)
output = open(str(out)-'.txt'+'modified.txt', 'w')
output.write(x)
output.close()