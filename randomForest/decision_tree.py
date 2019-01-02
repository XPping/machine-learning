from __future__ import division, print_function
import numpy as np



class TreeNodeStruct(object):
    """
    处理连续性数据，对于第i列特征，小于threshold则进入左子树，
    否则进入右子树。如果为孩子结点，则返回类别
    """
    def __init__(self, feature_i=None, threshold=None, left_substree=None, right_substree=None, tag=None):

        self.feature_i = feature_i
        self.threshold = threshold
        self.left_substree = left_substree
        self.right_substree = right_substree
        self.tag = tag

class DecisionTree(object):
    """
    # Classifier DecisionTree
    """
    def __init__(self, min_gain=1e-6, min_samples_in_node=2, max_depth=8):
        self.root = None
        self.min_gain = min_gain
        self.min_samples_in_node = min_samples_in_node
        self.max_depth = max_depth

    def _calEntropy(self, y):
        unqiue_y = np.unique(y)
        entropy = 0
        for u_y in unqiue_y:
            count = len(y[y==u_y]) + 0.0
            p = count / len(y)
            if p != 0:
                entropy += (-p * np.log2(p))
        return entropy

    def _calGain(self, y, left_y, right_y):
        p = len(left_y) / (len(y) + 0.0)

        entroy = self._calEntropy(y)

        gain = entroy - p * self._calEntropy(left_y) - (1-p)*self._calEntropy(right_y)

        return gain

    def _calTag(self, y):
        max_tag = 0
        max_tag_count = 0

        unique_y = np.unique(y)
        for u_y in unique_y:
            count = len(y[y==u_y])
            if count > max_tag_count:
                max_tag = u_y
                max_tag_count = count
        return max_tag

    def _buildTree(self, X, y, depth=0):
        print("tree depth: ", depth)
        X = X.reshape(len(X), -1)
        y = y.reshape(len(y), 1)
        samples_num, features_num = X.shape[0:2]

        if samples_num < self.min_samples_in_node or depth > self.max_depth:
            return TreeNodeStruct(tag=self._calTag(y))

        # The largest gain in current depth
        largest_gain = 0
        # The tree node correspond to the largest gain
        node = {}
        Xy = np.concatenate((X, y), axis=1)
        for feature_i in range(features_num):
            feature_values = X[:, feature_i]
            unique_values = np.unique(feature_values)
            for u_v in unique_values:
                Xy1 = Xy[Xy[:, feature_i]<u_v]
                Xy2 = Xy[Xy[:, feature_i]>=u_v]
                # Calculate gain
                gain = self._calGain(y, Xy1[:, features_num::], Xy2[:, features_num::])

                if gain > largest_gain:
                    largest_gain = gain
                    node["feature_i"] = feature_i
                    node["threshold"] = u_v
                    node["leftX"] = Xy1[:, 0:features_num]
                    node["lefty"] = Xy1[:, features_num::]
                    node["rightX"] = Xy2[:, 0:features_num]
                    node["righty"] = Xy2[:, features_num::]
        if largest_gain > self.min_gain:
            left_subtree = self._buildTree(node["leftX"], node["lefty"], depth+1)
            right_substree = self._buildTree(node["rightX"], node["righty"], depth+1)

            return TreeNodeStruct(feature_i=node["feature_i"], threshold=node["threshold"],
                                  left_substree=left_subtree, right_substree=right_substree, tag=None)
        tag = self._calTag(y)
        return TreeNodeStruct(tag=tag)

    def train(self, X, y):
        self.root = self._buildTree(X, y, depth=0)

    def predictOne(self, X, root=None):
        if root is None:
            root = self.root
        if root.tag is not None:
            return root.tag

        feature_i_value = X[root.feature_i]

        if feature_i_value < root.threshold:
            subtree = root.left_substree
        else:
            subtree = root.right_substree

        return self.predictOne(X, subtree)
    def predict(self, X):
        y_pred = []
        for x in X:
            y_pred.append(self.predictOne(x, root=None))
        return y_pred


class RegressionDecisionTree(object):
    """
    # Regression DecisionTree
    """
    def __init__(self, min_gain=1e-6, min_samples_in_node=2, max_depth=8):
        self.root = None
        self.min_gain = min_gain
        self.min_samples_in_node = min_samples_in_node
        self.max_depth = max_depth

    def _calEntropy(self, y):
        unqiue_y = np.unique(y)
        entropy = 0
        for u_y in unqiue_y:
            count = len(y[y==u_y]) + 0.0
            p = count / len(y)
            if p != 0:
                entropy += (-p * np.log2(p))
        return entropy

    def _calGain(self, y, left_y, right_y):
        p = len(left_y) / (len(y) + 0.0)

        entroy = self._calEntropy(y)

        gain = entroy - p * self._calEntropy(left_y) - (1-p)*self._calEntropy(right_y)

        return gain

    def _calTag(self, y):
        max_tag = 0
        max_tag_count = 0

        unique_y = np.unique(y)
        for u_y in unique_y:
            count = len(y[y==u_y])
            if count > max_tag_count:
                max_tag = u_y
                max_tag_count = count
        return max_tag

    def _buildTree(self, X, y, depth=0):
        print("tree depth: ", depth)
        X = X.reshape(len(X), -1)
        y = y.reshape(len(y), 1)
        samples_num, features_num = X.shape[0:2]

        if samples_num < self.min_samples_in_node or depth > self.max_depth:
            return TreeNodeStruct(tag=self._calTag(y))

        # The largest gain in current depth
        largest_gain = 0
        # The tree node correspond to the largest gain
        node = {}
        Xy = np.concatenate((X, y), axis=1)
        for feature_i in range(features_num):
            feature_values = X[:, feature_i]
            unique_values = np.unique(feature_values)
            for u_v in unique_values:
                Xy1 = Xy[Xy[:, feature_i]<u_v]
                Xy2 = Xy[Xy[:, feature_i]>=u_v]
                # Calculate gain
                gain = self._calGain(y, Xy1[:, features_num::], Xy2[:, features_num::])

                if gain > largest_gain:
                    largest_gain = gain
                    node["feature_i"] = feature_i
                    node["threshold"] = u_v
                    node["leftX"] = Xy1[:, 0:features_num]
                    node["lefty"] = Xy1[:, features_num::]
                    node["rightX"] = Xy2[:, 0:features_num]
                    node["righty"] = Xy2[:, features_num::]
        if largest_gain > self.min_gain:
            left_subtree = self._buildTree(node["leftX"], node["lefty"], depth+1)
            right_substree = self._buildTree(node["rightX"], node["righty"], depth+1)

            return TreeNodeStruct(feature_i=node["feature_i"], threshold=node["threshold"],
                                  left_substree=left_subtree, right_substree=right_substree, tag=None)
        tag = self._calTag(y)
        return TreeNodeStruct(tag=tag)

    def train(self, X, y):
        self.root = self._buildTree(X, y, depth=0)

    def predictOne(self, X, root=None):
        if root is None:
            root = self.root
        if root.tag is not None:
            return root.tag

        feature_i_value = X[root.feature_i]

        if feature_i_value < root.threshold:
            subtree = root.left_substree
        else:
            subtree = root.right_substree

        return self.predictOne(X, subtree)
    def predict(self, X):
        y_pred = []
        for x in X:
            y_pred.append(self.predictOne(x, root=None))
        return y_pred