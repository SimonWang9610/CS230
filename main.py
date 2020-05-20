import numpy as np

from sklearn.datasets import fetch_openml

from Models import build_model
from NeuralNetwork import NetworkLayers, MomentumNetwork, RMSNetwork, AdamNetwork

def scale_labels(Y, targtes):
    temp = np.zeros((targtes, Y.shape[1]))
    for i in range(Y.shape[1]):
        temp[int(Y[0, i]), i] = 1
    return temp

mnist = fetch_openml('mnist_784', version=1)

X, y = mnist['data'], mnist['target']


X_train, y_train = X[:10000,].T, y[:10000,].reshape(1, 10000)
y_train = scale_labels(y_train, len(np.unique(y)))

X_test, y_test = X[10000:12000,].T, y[10000:12000,].reshape(1, 2000)
y_test = scale_labels(y_test, len(np.unique(y)))

networks = [NetworkLayers(layer=1, neurons=200), NetworkLayers(layer=2, neurons=50),
            NetworkLayers(layer=3, neurons=10, final=True)]

build_model(X_train, y_train, X_test, y_test, 200, networks, 0.01, 128, model='gd', log_loss=True)

