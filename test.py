import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    temp = x - x.max()
    return np.exp(temp) / np.sum(np.exp(temp), axis=0, keepdims=True)

def tanh(x):
    a = np.exp(x)
    b = np.exp(-x)
    return (a - b) / (a + b)

def tanh2derv(x):
    return 1 - tanh(x)

def relu(x):
    return (x >= 0).astype(int) * x

def relu2deriv(x):
    return (x >= 0).astype(int)

def scale(x):
    for i in range(x.shape[1]):
        print(np.argmax(x[:,i:i+1]))


def scale_labels(Y, targets):
    temp = np.zeros((targets, Y.shape[1]))
    for i in range(Y.shape[1]):
        print(i)
        temp[int(Y[0, i]),i] = 1
        print(i)
    return temp

a = np.arange(0.000000001, 1, 0.01)
b = np.log(a)
plt.plot(a, b)
plt.show()