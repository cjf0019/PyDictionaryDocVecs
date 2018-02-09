# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 15:29:38 2018

NLTKRNTN is a modification of the rntn.py Recursive Neural Tensor Network code
by Patrick D. Smith found here: https://github.com/pdsmith1223 . This code differs
in that it 
1) Uses the nltk tree structure for inputs and labels. 
2) Accomodates cases where the number of leaves is not equal to two.
3) Uses a squared difference cost function for the prediction cost.
4) Allows for fixed embeddings for both the inputs and labels.

@author: InfiniteJest
"""

import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from nltk.tree import Tree

from sklearn.utils import shuffle

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

def get_output_recursivetest(tree, list_of_logits, is_root=False):
    if len(tree) == 1:
        if isinstance(tree, Tree):
            if isinstance(tree[0], Tree):    
                return get_output_recursivetest(tree[0], list_of_logits)
            else:
                x = np.array([1, 2, 3])
#            elif isinstance(tree, str):
#                index = self.word2index[tree]    #get word2vec model index for embedding lookup
#                print(tree, index)
#                x = tf.nn.embedding_lookup(self.We, index)
        else:
            return []
    elif len(tree) > 2:    #convert trees with more than two leaves into smaller trees
        for subtree in tree:
            if isinstance(subtree, Tree):
                if tree.index(subtree) == 0:
                    first = subtree
                else:
                    first = Tree(tree.label(), [first, subtree])
            else:
                pass
        return get_output_recursivetest(first, list_of_logits)
    elif len(tree) == 2:
        x1 = get_output_recursivetest(tree[0], list_of_logits, is_root=False)
        x2 = get_output_recursivetest(tree[1], list_of_logits, is_root=False)
        if x1 is None and x2 is None:
            return
        elif x1 is None and x2 is not None:
            x = x2
        elif x2 is None is x1 is not None:
            x = x1
        else:
            x = np.array([np.dot(x1, x2), 2, 1])
    
    list_of_logits.append(x)
    print(list_of_logits)
    return x

def get_labels(tree, doembed = False, tfembed = None):
    # must be returned in the same order as tree logits are returned
    # post-order traversal
    if len(tree) == 1:
        if isinstance(tree, Tree):
            if isinstance(tree[0], Tree):
                return get_labels(tree[0])
            else:
                print(doembed)
                if doembed == True:
                    print(tf.nn.embedding_lookup(tfembed, tree.label()))
                    return [tf.nn.embedding_lookup(tfembed, tree.label())]
                else:
                    print('here')
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
        if doembed == True:
            return get_labels(first) + [tf.nn.embedding_lookup(tfembed, tree.label())]
        else:    
            return get_labels(first) + [tree.label()]

    elif len(tree) == 2:
        print(doembed)
        if doembed == True:
            print(tf.nn.embedding_lookup(tfembed, tree.label()))
            return get_labels(tree[0]) + get_labels(tree[1]) + \
                    [tf.nn.embedding_lookup(tfembed, tree.label())]
        else:
            print('here :(')
            return get_labels(tree[0]) + get_labels(tree[1]) + [tree.label()]


class NLTKRNTN:
    def __init__(self, V, D, K, activation, word2vecmodel=None, word2index=None, 
                 index2word=None, fixembeddings=False):
        self.D = D       #hidden layer number of neurons
        self.f = activation
        self.embeddings = word2vecmodel   #embedding matrix (can be numpy array; the syn0 matrix in gensim word2vec)
        self.word2index = word2index    #dictionary to convert words into indices in embedding matrix
        self.index2word = index2word   #opposite dictionary of word2index
        self.includemb = False   #True if a word2vecmodel is supplied

        # word embedding
        if word2vecmodel.any() != None:
            print(self.embeddings)
            We = word2vecmodel  
            self.includemb = True
            if fixembeddings == True:   #make wordvectors/initial weights constants
                self.We = tf.get_variable("We", shape = np.shape(We), \
                        initializer=tf.constant_initializer(We, dtype=tf.float32))
            else:
                self.We = tf.get_variable("We", shape=np.shape(We),
                initializer=We.astype(np.float32))
        else:
            self.We = tf.get_variable("We", shape=np.shape(We),
            initializer=tf.contrib.layers.xavier_initializer())

        #Initialize all the weights
        self.W11 = tf.get_variable("W11", shape=(D, D, D), \
                                   initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        self.W22 = tf.get_variable("W22", shape=(D, D, D), \
                                   initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        self.W12 = tf.get_variable("W12", shape=(D, D, D), \
                                   initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        self.W1 = tf.get_variable("W1", shape=(D, D), \
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        self.W2 = tf.get_variable("W2", shape=(D, D), \
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        self.bh = tf.get_variable("bh", initializer=tf.zeros(D), dtype=tf.float32)
        self.Wo = tf.get_variable("Wo", shape=(D, D), \
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        self.bo = tf.get_variable("bo", initializer=tf.zeros(K), dtype=tf.float32)
        self.params = [self.We, self.W11, self.W22, self.W12, self.W1, self.W2, self.Wo]

    def fit(self, trees, lr=10e-3, mu=0.9, reg=10e-2, epochs=5):
        train_ops = []
        costs = []
        predictions = []
        all_labels = []
        i = 0
        N = len(trees)
        print("Compiling ops")
        for t in trees:
            i += 1
            sys.stdout.write("%d/%d\r" % (i, N))
            sys.stdout.flush()
            logits = self.get_output(t)
            
            labels = self.get_labels_with_emb(t)
            all_labels.append(labels)

            cost = self.get_cost(logits, labels, reg)
            costs.append(cost)

            prediction = tf.argmax(logits, 1)
            predictions.append(prediction)

            train_op = tf.train.MomentumOptimizer(lr, mu).minimize(cost)
            train_ops.append(train_op)

        # save for later so we don't have to recompile if we call score
        self.predictions = predictions
        self.all_labels = all_labels

        self.saver = tf.train.Saver()
        init = tf.initialize_all_variables()
        actual_costs = []
        per_epoch_costs = []
        correct_rates = []
        with tf.Session() as session:
            session.run(init)

            for i in xrange(epochs):
                train_ops, costs, predictions, all_labels = shuffle(train_ops, costs, predictions, all_labels)
                epoch_cost = 0
                n_correct = 0
                n_total = 0
                j = 0
                N = len(train_ops)
                for train_op, cost, prediction, labels in zip(train_ops, costs, predictions, all_labels):
                    _, c, p = session.run([train_op, cost, prediction])
                    epoch_cost += c
                    actual_costs.append(c)
                    n_correct += np.sum(p == labels)
                    n_total += len(labels)

                    j += 1
                    if j % 10 == 0:
                        sys.stdout.write("j: %d, N: %d, c: %f\r" % (j, N, c))
                        sys.stdout.flush()

                    if np.isnan(c):
                        exit()

                per_epoch_costs.append(epoch_cost)
                correct_rates.append(n_correct / float(n_total))

            self.save_path = self.saver.save(session, "tf_model.ckpt")

        plt.plot(actual_costs)
        plt.title("cost per train_op call")
        plt.show()

        plt.plot(per_epoch_costs)
        plt.title("per epoch costs")
        plt.show()

        plt.plot(correct_rates)
        plt.title("correct rates")
        plt.show()

    def get_cost(self, logits, labels, reg):
        cost = tf.reduce_mean(tf.squared_difference(logits, labels))
        rcost = sum(tf.nn.l2_loss(p) for p in self.params)
        cost += reg*rcost
        return cost

    # list_of_logits is an output!
    # it is added to using post-order traversal
    def get_labels_with_emb(self, tree):
        # must be returned in the same order as tree logits are returned
        # post-order traversal
        if len(tree) == 1:
            if isinstance(tree, Tree):
                if isinstance(tree[0], Tree):
                #essentially labels of labels, so go next leaf down    
                    return self.get_labels_with_emb(tree[0])   
                else:
                    return [tf.nn.embedding_lookup(self.embeddings, tree.label())]  #return label as embedding
            else:
                return []

        elif len(tree) > 2:
            #we want to break down these trees so they have only two leaves
            for subtree in tree:
#                if isinstance(subtree, Tree):
                if tree.index(subtree) == 0:   #first subtree
                    first = subtree
                else:
                    first = Tree(tree.label(), [first, subtree])  #make subtree to first tree
#                else:
#                    pass
#            return self.get_labels_with_emb(first) + [tf.nn.embedding_lookup(self.embeddings, tree.label())]
            return self.get_labels_with_emb(first)

        elif len(tree) == 2:
            return self.get_labels_with_emb(tree[0]) + self.get_labels_with_emb(tree[1]) + \
                    [tf.nn.embedding_lookup(self.embeddings, tree.label())]

    def get_output_recursive(self, tree, list_of_logits, is_root=False):
        if len(tree) == 1:
            if isinstance(tree, Tree):
                if isinstance(tree[0], Tree):    
                    return self.get_output_recursive(tree[0], list_of_logits)
                else:
                    index = self.word2index[tree.leaves()[0]]
                    x = tf.nn.embedding_lookup(self.We, index)
                    x = tf.reshape(x, (-1, self.D)) 
#            elif isinstance(tree, str):
#                index = self.word2index[tree]    #get word2vec model index for embedding lookup
#                print(tree, index)
#                x = tf.nn.embedding_lookup(self.We, index)
            else:
                return []
        elif len(tree) > 2:    #convert trees with more than two leaves into smaller trees
            for subtree in tree:
#                if isinstance(subtree, Tree):
                if tree.index(subtree) == 0:
                    first = subtree
                else:
                    first = Tree(tree.label(), [first, subtree])
#                else:
#                    pass
            return self.get_output_recursive(first, list_of_logits)   #iterate back through with new tree structure
        elif len(tree) == 2:
            x1 = self.get_output_recursive(tree[0], list_of_logits, is_root=False)
            x2 = self.get_output_recursive(tree[1], list_of_logits, is_root=False)
            if x1 is None and x2 is None:
                return
            elif x1 is None and x2 is not None:
                x = x2
            elif x2 is None is x1 is not None:
                x = x1
            else:
                x = self.f(
                    tensor_mul(self.D, x1, self.W11, x1) +
                    tensor_mul(self.D, x2, self.W22, x2) +
                    tensor_mul(self.D, x1, self.W12, x2) +
                    tf.matmul(x1, self.W1) +
                    tf.matmul(x2, self.W2) +
                    self.bh)
    
        x = tf.cast(x, dtype=tf.float32)
        logits = tf.matmul(x, self.Wo) + self.bo
        list_of_logits.append(logits)
        return x

    def get_output(self, tree):
        logits = []
        try:
            self.get_output_recursive(tree, logits)
        except Exception as e:
            raise e
        return tf.concat(0, logits)

    def score(self, trees):
        if trees is None:
            predictions = self.predictions
            all_labels = self.all_labels
        else:
            # just build and run the predict_op for each tree
            # and accumulate the total
            predictions = []
            all_labels = []

            i = 0
            N = len(trees)
            print("Compiling ops")
            for t in trees:

                i += 1
                sys.stdout.write("%d/%d\r" % (i, N))
                sys.stdout.flush()

                logits = self.get_output(t)
                labels = self.get_labels_with_emb(t)
                all_labels.append(labels)

                prediction = tf.argmax(logits, 1)
                predictions.append(prediction)

        n_correct = 0
        n_total = 0
        with tf.Session() as session:
            self.saver.restore(session, "tf_model.ckpt")

            for prediction, y in zip(predictions, all_labels):
                p = session.run(prediction)
                # print "pred:", p
                # print "label:", y
                # n_correct += np.sum(p == y)
                n_correct += (p[-1] == y[-1]) # we only care about the root
                n_total += len(y)

        return float(n_correct) / n_total

