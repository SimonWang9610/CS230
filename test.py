import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)

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
drop = 0.5
a = np.random.rand(3, 4)
b = (a > drop).astype(int)
print(b)


