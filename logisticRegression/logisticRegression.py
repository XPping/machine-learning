from __future__ import division, print_function

import numpy as np
np.random.seed(12344)
class LogisticRegression(object):
    def __init__(self, N, alpha, step=1000):
        self.theta = np.ones((N, 1))
        self.alpha = alpha
        self.step = step

    def sigmoid(self, X):
        return 1.0 / (1 + np.exp(-X))

    def h(self, X):
        h = self.theta.T * X
        h = h.sum(axis=1)
        return self.sigmoid(h)

    def train(self, X, y):
        for i in range(self.step):
            acc = self.predict(X, y)
            print("The accuracy of {}-th iter is {}.".format(i, acc))
            if 1 - acc < 0.01:
                break
            predict = self.h(X)
            # The gradient of theta
            theta_grad = np.sum(X * (predict - y).reshape(-1, 1), axis=0).reshape(-1, 1)
            self.theta -= (self.alpha * theta_grad)
            # break

    def predict(self, X, y=None):
        pred = self.h(X)
        pred = pred >= 0.5
        pred = np.where(pred, 1, 0)
        if y is None:
            return pred
        # print(pred, "ha", y)
        # print(pred.shape, y.shape)
        acc = np.equal(pred, y).sum() + 0.0
        return acc / len(X)

