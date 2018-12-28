from __future__ import division, print_function

import os
import numpy as np

class SVM(object):
    def __init__(self, kernel_type='linear', C=1.0, sigma=0.5, epsilon=1e-5, max_iter=1000):
        self.kernel = self.linearKernel
        if kernel_type == 'gaussian':
            self.kernel = self.gaussianKernel
        self.C = C
        self.sigma = sigma
        self.epsilon = epsilon
        self.max_iter = max_iter

    def linearKernel(self, x1, x2):
        return np.dot(x1, x2)

    def gaussianKernel(self, x1, x2):
        return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * (self.sigma ** 2)))

    def sign(self, X, w, b):
        return np.sign(np.dot(w.T, X.T) + b).astype(int)

    def calcE(self, X, y, w, b):
        return self.sign(X, w, b) - y

    def calcBoarder(self, alpha_i, alpha_j, y_i, y_j):
        if y_i == y_j:
            L = max(0.0, alpha_i + alpha_j - self.C)
            H = min(self.C, alpha_i + alpha_j)
        else:
            L = max(0.0, alpha_j - alpha_i)
            H = min(self.C, self.C + alpha_j - alpha_i)
        return L, H

    def getIndexExceptJ(self, N, j):
        ret = j
        while ret == j:
            ret = np.random.randint(0, N-1)
        return ret

    def calcW(self, alpha, X, y):
        return np.dot(alpha * y, X)

    def calcb(self, X, y, w):
        b = y - np.dot(w.T, X.T)
        return np.mean(b)

    def train(self, X, y):
        N = X.shape[0]
        D = X.shape[1]
        alpha = np.zeros((N,), dtype=np.float32)
        for iter in range(self.max_iter):
            pre_alpha = np.copy(alpha)
            for j in range(N):
                i = self.getIndexExceptJ(N, j)
                xi, xj = X[i], X[j]
                yi, yj = y[i], y[j]
                alpha_i, alpha_j = alpha[i], alpha[j]
                # Eta in func
                Eta = self.kernel(xi, xi) + self.kernel(xj, xj) - 2 * self.kernel(xi, xj)
                if Eta == 0:
                    continue
                # Update weight hypermaters
                self.w = self.calcW(alpha, X, y)
                self.b = self.calcb(X, y, self.w)
                # Ei and Ej
                Ei = self.calcE(xi, yi, self.w, self.b)
                Ej = self.calcE(xj, yj, self.w, self.b)
                # L, H
                L, H = self.calcBoarder(alpha_i, alpha_j, yi, yj)
                # Update alpha[i] and alpha[j]
                alpha_j_new = alpha_j + yj * (Ei - Ej) / Eta
                alpha_j_new = min(H, alpha_j_new)
                alpha_j_new = max(L, alpha_j_new)
                alpha_i_new = alpha_i + yi * yj * (alpha_j - alpha_j_new)
                alpha[i] = alpha_i_new
                alpha[j] = alpha_j_new
            # Check convergence
            diff = np.linalg.norm(alpha - pre_alpha)
            print("iter: {}, diff: {}".format(iter, diff))
            if diff < self.epsilon:
                break
        self.b = self.calcb(X, y, self.w)
        self.w = self.calcW(alpha, X, y)

    def predict(self, X):
        return self.sign(X, self.w, self.b)
