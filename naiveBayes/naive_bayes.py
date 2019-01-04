from __future__ import division, print_function
import numpy as np


class NaiveBayes(object):
    def __init__(self):
        pass

    def train(self, X, y):
        X = X.reshape(len(X), -1)
        y = y.reshape(len(y), 1)
        self.classes = np.unique(y)
        self.feature_mean_var = []
        for i, label in enumerate(self.classes):
            X_i = X[np.where(y == label)[0]]
            mean = np.mean(X_i, axis=0)
            var = np.var(X_i, axis=0)
            self.feature_mean_var.append((mean, var))

    def _gaussian(self, X, mean, var):
        eps = 1e-4
        ret = 1.0 / np.sqrt(2 * np.pi * var + eps) * np.exp(-np.power(X-mean, 2) / (2.0 * var + eps))
        ret = np.log(ret + eps)
        ret = np.sum(ret, axis=1)
        return ret

    def predict(self, X):
        y_red = np.zeros((len(X), len(self.classes)), dtype=np.float32)
        for i, label in enumerate(self.classes):
            mean, var = self.feature_mean_var[i]
            predict = self._gaussian(X, mean, var)
            y_red[:, i] = predict
        y_pred = np.argmax(y_red, axis=1)
        return y_pred