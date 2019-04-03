import numpy as np
from sklearn.datasets import fetch_openml
import os

from network import ConvNet
from solver import Solver
from layers import *

def train():
    # load data
    os.chdir("p1/data/")
    X = np.load('../data/x.npy')
    y = np.load('../data/y.npy')
    y = y.reshape(y.shape[0],1)
    n, d = X.shape
    print(X.shape)
    n, m = y.shape
    print(y.shape)

    w = np.random.randn(d, m)
    b = np.random.randn(m)

    learning_rate = 0.1
    
    for i in range(10000):
        y_pred, cache = fc_forward(X, w, b)
        loss, dout = l2_loss(y_pred, y)
        if i % 50 == 0:
            print(loss)
        dx, dw, db = fc_backward(dout, cache)
        # learning_rate = 1 / (i + 1)
        w -= dw * learning_rate
        b -= db * learning_rate
        
        
     

if __name__=="__main__":
    train()
