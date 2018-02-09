# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 15:51:31 2018
Used to extract parse trees from Stanford's CoreNLP Constituency Parser. After 
running the parser, this code extracts the trees and places them in a modified 
version of the NLTK tree, called "NLTKTreeNoPunct," which was used to discard 
punctuation parses. 
@author: InfiniteJest
"""
import os
import re
os.chdir('C:\\Users\\InfiniteJest\\Documents\\Python_Scripts')
from nltk.tree import Tree
from six import string_types
import tensorflow as tf

test = '(ROOT  (NP    (NP (NN monopoly))    (: :)    (NP      (NP (JJ exclusive) (NN control)        (CC or)        (NN possession))      (PP (IN of)        (NP (NN something))))    (. .)))'
test2 = """(ROOT
  (NP
    (NP
      (ADJP (JJ alive))
      (NNS means))
    (ADJP (RB mentally) (JJ perceptive)
      (CC and)
      (JJ responsive))
    (. .)))"""
      
    
def get_labels(tree, tfembed = None, label2index = None):
    # must be returned in the same order as tree logits are returned
    # post-order traversal
    if len(tree) == 1:
        if isinstance(tree, Tree):
            if isinstance(tree[0], Tree):
                return get_labels(tree[0])
            else:
                if tfembed != None and label2index != None:
                    return tf.nn.embedding_lookup(tfembed, label2index[tree.label()])
                elif tfembed != None and label2index == None:
                    print("Must specify a label2index dictionary for embeddings!")
                else:
                    return [tree.label()]
        else:
            return []
    elif len(tree) > 2:
        for subtree in tree:
            if isinstance(subtree, Tree):
                if tree.index(subtree) == 0:
                    first = subtree
                else:
                    first = Tree(tree.label(), [first, subtree])
            else:
                pass
        return get_labels(first) + [tree.label()]
    elif len(tree) == 2:
        return get_labels(tree[0]) + get_labels(tree[1]) + [tree.label()]



class NLTKTreeNoPunct(Tree):
    def __init__(self, node, children=None):
        super(NLTKTreeNoPunct, self).__init__(node=node, children=children)
        
    @classmethod
    def fromstringnopunct(cls, s, brackets='()', read_node=None, read_leaf=None,
              node_pattern=None, leaf_pattern=None,
              remove_empty_top_bracketing=False):
        """
        This is a modification of the "fromstring" function of the nltk Tree.
        The original source code can be found here: 
        
            http://www.nltk.org/_modules/nltk/tree.html.
            
        This function extracts the parsed tree from a text file, but also 
        removes the period from the tree.
        
        """
        if not isinstance(brackets, string_types) or len(brackets) != 2:
            raise TypeError('brackets must be a length-2 string')
        if re.search('\s', brackets):
            raise TypeError('whitespace brackets not allowed')
        #Remove the period tagging.
        s = re.sub('(\(\. \.\))|(\(\: \:\))|(\(\`\` \`\))|(\(\'\' \'\))|(\(\: \-\))', '', s)
        # Construct a regexp that will tokenize the string.
        open_b, close_b = brackets
        open_pattern, close_pattern = (re.escape(open_b), re.escape(close_b))
        if node_pattern is None:
            node_pattern = '[^\s%s%s]+' % (open_pattern, close_pattern)
        if leaf_pattern is None:
            leaf_pattern = '[^\s%s%s]+' % (open_pattern, close_pattern)
        token_re = re.compile('%s\s*(%s)?|%s|(%s)' % (
            open_pattern, node_pattern, close_pattern, leaf_pattern))
        # Walk through each token, updating a stack of trees.
        stack = [(None, [])] # list of (node, children) tuples
        for match in token_re.finditer(s):
            punctuation = [".", "!", "?", ",", "'"]
            token = match.group()
            if token in punctuation:
                pass
            else:
                # Beginning of a tree/subtree
                if token[0] == open_b:
                    if len(stack) == 1 and len(stack[0][1]) > 0:
                        cls._parse_error(s, match, 'end-of-string')
                    label = token[1:].lstrip()
                    if read_node is not None: label = read_node(label)
                    stack.append((label, []))
                # End of a tree/subtree
                elif token == close_b:
                    if len(stack) == 1:
                        if len(stack[0][1]) == 0:
                            cls._parse_error(s, match, open_b)
                        else:
                            cls._parse_error(s, match, 'end-of-string')
                    label, children = stack.pop()
                    stack[-1][1].append(cls(label, children))
                # Leaf node
                else:
                    if len(stack) == 1:
                        cls._parse_error(s, match, open_b)
                    if read_leaf is not None: token = read_leaf(token)
                    stack[-1][1].append(token)

        # check that we got exactly one complete tree.
        if len(stack) > 1:
            cls._parse_error(s, 'end-of-string', close_b)
        elif len(stack[0][1]) == 0:
            cls._parse_error(s, 'end-of-string', open_b)
        else:
            assert stack[0][0] is None
            assert len(stack[0][1]) == 1
        tree = stack[0][1][0]

        # If the tree has an extra level with node='', then get rid of
        # it.  E.g.: "((S (NP ...) (VP ...)))"
        if remove_empty_top_bracketing and tree._label == '' and len(tree) == 1:
            tree = tree[0]
        # discard the tree if it is of a punctuation
        return tree




def xtrcttree(file):
    firstline = file.readline()
    numtokens = re.search('(?<=[\(])[0-9]+(?= token)', firstline)
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
            file.readline()
        else:
            tree.append(treeline)
    return ''.join(tree)


def add_embedding(self, V, D):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:
            - Creates an embedding tensor and initializes it with self.pretrained_embeddings.
            - Uses the input_placeholder to index into the embeddings tensor, resulting in a
              tensor of shape (None, n_features, embedding_size).
            - Concatenates the embeddings by reshaping the embeddings tensor to shape
              (None, n_features * embedding_size).

        Hint: You might find tf.nn.embedding_lookup useful.
        Hint: You can use tf.reshape to concatenate the vectors. See following link to understand
            what -1 in a shape means.
            https://www.tensorflow.org/api_docs/python/array_ops/shapes_and_shaping#reshape.

        Returns:
            embeddings: tf.Tensor of shape (None, n_features*embed_size)
        """
        ### YOUR CODE HERE
    emb = tf.placeholder(tf.float32, shape=(V, D))
    embeddings = tf.nn.embedding_lookup(emb, self.input_placeholder)
    embeddings = tf.reshape(embeddings, (-1, self.config.n_features*self.config.embed_size))
        ### END YOUR CODE
    return embeddings


def get_labels3(tree, tfembed = None):
    # must be returned in the same order as tree logits are returned
    # post-order traversal
    if len(tree) == 1:
        if isinstance(tree, Tree):
            if isinstance(tree[0], Tree):
                return get_labels(tree[0])
            else:
                if tfembed != None:
                    return [tf.nn.embedding_lookup(tfembed, tree.label())]
                else:
                    return [tree.label()]
        else:
            return []

    elif len(tree) > 2:
        for subtree in tree:
            if isinstance(subtree, Tree):
                if tree.index(subtree) == 0:
                    first = subtree
                else:
                    first = Tree(tree.label(), [first, subtree])
            else:
                pass
        if tfembed != None:
            return get_labels(first) + [tf.nn.embedding_lookup(tfembed, tree.label())]
        else:    
            return get_labels(first) + [tree.label()]

    elif len(tree) == 2:
        if tfembed != None:
            return get_labels(tree[0]) + get_labels(tree[1]) + \
                    [tf.nn.embedding_lookup(tfembed, tree.label())]
        else:
            return get_labels(tree[0]) + get_labels(tree[1]) + [tree.label()]


def get_labels1(tree):
    # must be returned in the same order as tree logits are returned
    # post-order traversal
    if len(tree) == 1:
        if isinstance(tree, Tree):
            if isinstance(tree[0], Tree):
                return get_labels(tree[0])
            else:
                return [tree.label()]
        else:
            return []
    elif len(tree) > 2:
        for subtree in tree:
            if isinstance(subtree, Tree):
                if tree.index(subtree) == 0:
                    first = subtree
                else:
                    first = Tree(tree.label(), [first, subtree])
            else:
                pass
        return get_labels(first) + [tree.label()]
    elif len(tree) == 2:
        return get_labels(tree[0]) + get_labels(tree[1]) + [tree.label()]
    
    

def process_branch(branch, keep_pos=False):
    branched = re.findall('(?<=\().*(?=\()', branch)
    if keep_pos==True:
        pos = re.search('[A-Z]*', branched.group())
        return branched, pos
    else:
        return branched
    

def process_tree(tree, keep_pos=False):
    treed = []
    partospeech = []
    branched = re.findall('(?<=\().*(?=\()', tree)
    treed.append(branched)
    if keep_pos == True:
        partospeech.append(re.search('[A-Z]*', branched.group()))
    while len(branched) != 0:
        if keep_pos == True:
            branched, pos = process_branch(branched[0], keep_pos=keep_pos)
            treed.append(branched)
            partospeech.append(pos)
        else:
            branched = process_branch(branched[0], keep_pos=keep_pos)
            treed.append(branched)
    if keep_pos == True:
        return treed, partospeech
    else:
        return treed