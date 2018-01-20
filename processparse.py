# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 15:51:31 2018

@author: InfiniteJest
"""
import os
import re
os.chdir('C:\\Users\\InfiniteJest\\Documents\\Python_Scripts')
import rntn
os.chdir('C:\\Users\\InfiniteJest\\Documents\\Python_Scripts\\stanford-corenlp-full-2017-06-09')
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
      
def get_labels(tree):
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
    

def tensor_mul(d, x1, A, x2):
    A = tf.reshape(A, [d, d*d])
    # (1 x d) x (d x dd)
    tmp = tf.matmul(x1, A)
    # (1 x dd)
    tmp = tf.reshape(tmp, [d, d])
    # (d x d)
    tmp = tf.matmul(tmp, tf.transpose(x2))
    # (d x 1)
    return tf.reshape(tmp, [1, d])


class NLTKRNTN(RNTN):
    def __init__(self, V, D, K, activation):
        super(RNTN, self).__init__(V, D, K, activation)
        
    def get_output_recursive(self, tree, list_of_logits):
        if len(tree) == 1:
            if isinstance(tree, Tree):
                    x = self.get_output_recursive(tree, list_of_logits)
            elif isinstance(tree, str):
                x = tf.nn.embedding_lookup(self.We, [tree])
        elif len(tree) > 2:
            for subtree in tree:
                if isinstance(subtree, Tree):
                    if tree.index(subtree) == 0:
                        first = subtree
                    else:
                        first = Tree(tree.label(), [first, subtree])
                else:
                    pass
            return self.get_output_recursive(first, list_of_logits)
        elif len(tree) == 2:
            x1 = self.get_output_recursive(tree[0], list_of_logits, is_root=False)
            x2 = self.get_output_recursive(tree[1], list_of_logits, is_root=False)
            x = self.f(
                tensor_mul(self.D, x1, self.W11, x1) +
                tensor_mul(self.D, x2, self.W22, x2) +
                tensor_mul(self.D, x1, self.W12, x2) +
                tf.matmul(x1, self.W1) +
                tf.matmul(x2, self.W2) +
                self.bh)
    
        logits = tf.matmul(x, self.Wo) + self.bo
        list_of_logits.append(logits)
        return x

class NLTKTreeNoPeriod(Tree):
    def __init__(self, node, children=None):
        super(NLTKTreeNoPeriod, self).__init__(node=node, children=children)
        
    @classmethod
    def fromstringnoperiod(cls, s, brackets='()', read_node=None, read_leaf=None,
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




class FileIterator(object):
    def __init__(self, file):
        self.file = file
    
    def __iter__(self):
        for i in open(self.file):
            yield i

