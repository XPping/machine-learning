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


class RegressionDecisionTree(object):
    """
    # Regression DecisionTree
    """
    def __init__(self, kethe_hyper=0., lambda_hyper=0., min_gain=1e-6, min_samples_in_node=2, max_depth=8):
        self.root = None
        self.kethe_hyper = kethe_hyper
        self.lambda_hyper = lambda_hyper
        self.min_gain = min_gain
        self.min_samples_in_node = min_samples_in_node
        self.max_depth = max_depth

    def _oneGradient(self, y, pred):
        """
        The first gradient of mean-square-error: 1/2(y-pred)^2
        :param y:
        :param pred:
        :return:
        """
        return y - pred

    def _twoGradient(self, y, pred):
        """
        The second gradient of mean-square-error: 1/2(y-pred)^2
        :param y:
        :param pred:
        :return:
        """
        return np.ones_like(y)

    def _gain(self, y, y_pred):
        G = np.power(self._oneGradient(y, y_pred).sum(), 2)
        H = self._twoGradient(y, y_pred).sum()
        # print(y.reshape(-1), y_pred.reshape(-1))
        # print(G, H)
        return G / (H + self.lambda_hyper)

    def _calGain(self, y, y_pred, left_y, left_y_pred, right_y, right_y_pred):
        A = self._gain(y, y_pred)
        B = self._gain(left_y, left_y_pred)
        C = self._gain(right_y, right_y_pred)
        # print(A, B, C)
        return 0.5 * (B + C - A - self.kethe_hyper)

    def _calTag(self, y, y_pred):
        A = np.sum(self._oneGradient(y, y_pred), axis=0)
        B = np.sum(self._twoGradient(y, y_pred), axis=0)
        return A / B

    def _buildTree(self, X, y, y_pred, depth=0):
        print("tree depth: ", depth)
        X = X.reshape(len(X), -1)
        y = y.reshape(len(y), 1)
        y_pred = y_pred.reshape(len(y_pred), 1)
        samples_num, features_num = X.shape[0:2]

        if samples_num < self.min_samples_in_node or depth > self.max_depth:
            return TreeNodeStruct(tag=self._calTag(y, y_pred))
        # The largest gain in current depth
        largest_gain = 0
        # The tree node correspond to the largest gain
        node = {}
        Xy = np.concatenate((X, y), axis=1)
        Xyy = np.concatenate((Xy, y_pred), axis=1)
        for feature_i in range(features_num):
            feature_values = X[:, feature_i]
            unique_values = np.unique(feature_values)
            for u_v in unique_values:
                Xyy1 = Xyy[Xyy[:, feature_i]<u_v]
                Xyy2 = Xyy[Xyy[:, feature_i]>=u_v]
                # Calculate gain
                gain = self._calGain(y, y_pred,
                                     Xyy1[:, features_num:(features_num+1)], Xyy1[:, (features_num+1)::],
                                     Xyy2[:, features_num:(features_num+1)], Xyy2[:, (features_num+1)::])

                if gain > largest_gain:
                    largest_gain = gain
                    node["feature_i"] = feature_i
                    node["threshold"] = u_v
                    node["leftX"] = Xyy1[:, 0:features_num]
                    node["lefty"] = Xyy1[:, features_num::]
                    node["rightX"] = Xyy2[:, 0:features_num]
                    node["righty"] = Xyy2[:, features_num::]
        if largest_gain > self.min_gain:
            left_subtree = self._buildTree(node["leftX"], node["lefty"][:, 0:1], node["lefty"][:, 1:2], depth+1)
            right_substree = self._buildTree(node["rightX"], node["righty"][:, 0:1], node["righty"][:, 1:2], depth+1)

            return TreeNodeStruct(feature_i=node["feature_i"], threshold=node["threshold"],
                                  left_substree=left_subtree, right_substree=right_substree, tag=None)
        tag = self._calTag(y, y_pred)
        return TreeNodeStruct(tag=tag)

    def train(self, X, y, y_pred):
        self.root = self._buildTree(X, y, y_pred, depth=0)

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
        return np.array(y_pred, dtype=np.float32)

class XGBoost(object):
    def __init__(self, tree_num=10, kethe_hyper=0., lambda_hyper=0., min_gain_in_tree=1e-6,
                 min_samples_in_tree_node=2, max_depth_in_tree=8):
        self.tree_num = tree_num
        self.kethe_hyper = kethe_hyper
        self.lambda_hyper = lambda_hyper
        self.min_gain_in_tree = min_gain_in_tree
        self.min_samples_in_tree_node = min_samples_in_tree_node
        self.max_depth_in_tree = max_depth_in_tree

        self.forest = []

    def train(self, X, y):
        X = X.reshape(len(X), -1)
        y = y.reshape(len(y), 1)
        y_pred = np.zeros_like(y, dtype=np.float32)
        # print(y.reshape(-1))
        for i in range(self.tree_num):
            # print(y_pred.reshape(-1))
            print("Build {}-th tree.".format(i+1))
            tree = RegressionDecisionTree(kethe_hyper=self.kethe_hyper,
                                          lambda_hyper=self.lambda_hyper,
                                          min_gain=self.min_gain_in_tree,
                                          min_samples_in_node=self.min_samples_in_tree_node,
                                          max_depth=self.max_depth_in_tree)
            tree.train(X, y, y_pred)
            self.forest.append(tree)
            pred = tree.predict(X)
            pred = pred.reshape(-1, 1)
            y_pred += pred

    def predict(self, X):
        y_pred = np.zeros(len(X))
        y_pred = y_pred.reshape(-1, 1)
        for tree in self.forest:
            pred = tree.predict(X)
            pred = pred.reshape(-1, 1)
            y_pred += pred
        return y_pred
