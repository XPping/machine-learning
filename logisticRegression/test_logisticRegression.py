from __future__ import division, print_function

import os
import shutil
import numpy as np
import cv2
from logisticRegression import LogisticRegression

def loadData(path):
    """
    The path is dir of mnist
    0/1.jpg, 0/2.jpg, ...
    1/1.jpg, 0/2.jpg, ...
    """
    X = []
    y = []
    dir1 = os.listdir(path)
    for d1 in dir1:
        dir2 = os.listdir(path+'/'+d1)
        for d2 in dir2:
            if int(d1) == 0:
                image = cv2.imread(path+r'/'+d1+r'/'+d2, 0)
                X.append(np.array(image, dtype=np.float32).reshape(-1) / 255.0)
                y.append(0)
            elif int(d1) == 1:
                image = cv2.imread(path+r'/'+d1+r'/'+d2, 0)
                X.append(np.array(image, dtype=np.float32).reshape(-1) / 255.0)
                y.append(1)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    perm = np.random.permutation(X.shape[0])
    X = X[perm]
    y = y[perm]
    return X, y

def test():
    X, y = loadData(r'../mnist')
    print(X.shape, y.shape)
    trainX, trainY = X[0:500], y[0:500]
    valX, valY = X[500:600], y[500:600]
    cls = LogisticRegression(N=len(trainX[0]), alpha=0.01, step=1000)
    cls.train(trainX, trainY)
    predict = cls.predict(valX)
    # print(predict)
    acc = 0.0
    for i in range(len(predict)):
        # print(predict[i], valY[i])
        if int(predict[i]) == int(valY[i]):
            acc += 1
    print("Acc: {}".format(acc / len(predict)))


if __name__ == "__main__":
    test()
