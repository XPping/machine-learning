from __future__ import division, print_function

import numpy as np
from decision_tree import DecisionTree

class RandomForest(object):
    def __init__(self, tree_num=10, min_gain_in_tree=1e-6, min_samples_in_tree_node=2, max_depth_in_tree=8):
        self.tree_num = tree_num
        self.min_gain_in_tree = min_gain_in_tree
        self.min_samples_in_tree_node = min_samples_in_tree_node
        self.max_depth_in_tree = max_depth_in_tree



        self.forest = []                # All trees
        self.forest_features_id = []    # The chosen features in per tree
        self.classes = None                 # The classes from 0 to number_classes

    def _generateSamples(self, N):
        """
        random choice [2*N/3, N] samples
        :param N: is the number of samples
        :return:
        """
        perm = np.random.permutation(N)
        num = np.random.randint(int(2*N/3), N)
        return perm[0:num]

    def _generateFeatures(self, M):
        """
        random choice [np.sqrt(M), M] features
        :param M: is the number of features
        :return:
        """
        perm = np.random.permutation(M)
        num = np.random.randint(int(np.sqrt(M)), M)
        return perm[0:num]

    def train(self, X, y):
        X = X.reshape(len(X), -1)
        y = y.reshape(len(y), 1)
        self.classes = np.unique(y)
        for i in range(self.tree_num):
            print("Build {}-th tree.".format(i+1))
            tree = DecisionTree(min_gain=self.min_gain_in_tree,
                                min_samples_in_node=self.min_samples_in_tree_node,
                                max_depth=self.max_depth_in_tree)
            s_perm = self._generateSamples(len(X))
            f_perm = self._generateFeatures(len(X[0]))
            _X = X[s_perm]
            _X = _X[:, f_perm]
            _y = y[s_perm]
            print(_X.shape, _y.shape)
            tree.train(_X, _y)
            self.forest.append(tree)
            self.forest_features_id.append(f_perm)

    def predict(self, X):
        y_pred = np.zeros((len(X), len(self.classes)))
        for i in range(self.tree_num):
            tree = self.forest[i]
            feature_id = self.forest_features_id[i]
            pred = tree.predict(X[:, feature_id])
            for id in range(len(pred)):
                y_pred[id, int(pred[id])] += 1
        y_pred = y_pred.argmax(axis=1)
        return y_pred

