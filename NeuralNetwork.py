import numpy as np


def relu(x):
    return (x >= 0).astype(int) * x

def relu_derivate(x):
    return (x >= 0).astype(int)

def tanh(x):
    return (np.exp(x) + np.exp(-x)) / (np.exp(x) + np.exp(-x))

def tanh_derivate(x):
    return 1 - tanh(x) ** 2

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)

def softmax_derivate(x):
    pass

class NetworkLayers(object):

    def __init__(self, layer, neurons, activation, reg='l1', lambd=0, drop=0, final=False):
        '''
        :param layer: i-th layer, integer
        :param neurons: number of nodes of current layer, integer
        :param activation: type of activation functions, str: 'tanh' or 'relu'
        :param reg: regularization, default 'l1', string
        :param lambd: default '0' is no regularization, float
        :param drop: percentage of neurons dropped, float
        :param final: whether the layer is output layer

        attributes:
        :attr weights: weights
        :attr bias: bias
        :attr input: the activated output of previous layer, matrix, shape(self.neurons, prev.neurons)
        :attr output: the activated output, matrix, shape(self.neurons, prev,neurons)
        :attr derivates: caches for backward_propagation, include: [dA, dZ, dW] (db=dZ)
        '''
        self.layer = layer
        self.neurons = neurons
        self.activation = activation
        self.reg = reg
        self.lambd = lambd
        self.drop = drop
        self.final = final

        self.weights = None
        self.bias = None

        self.input = None
        self.output = None

        self.derivates = []

    def initialize_weights_bias(self, col):
        '''
        :param col: prev.neurons
        :return: weights initialized according to Xavier or He Initialization
        '''
        if self.weights is None:

            if self.activation =='tanh':
                scale = np.sqrt(1 / col)
            else:
                scale = np.sqrt(2 / col)

            self.weights = np.random.randn(self.neurons, col) * scale
            self.bias = np.zeros((self.neurons, 1))

        return self

    def calculate_input_output(self, prev_output):
        '''
        :param prev_output:
        :return: input and output of current layer
        '''

        self.input = self.weights.T.dot(prev_output) + self.bias

        # add dropout
        if self.drop:
            temp = np.random.rand(self.input.shape)
            mask = (temp > self.drop).astype(int)
            self.input *= mask / (1 - self.drop)

        if not self.final:
            if self.activation == 'tanh':
                self.output = tanh(self.input)
            else:
                self.output = relu(self.input)
        else:
            self.output = softmax(self.input)


    def final_derivates(self, Y):
        '''
        :param Y: reshaped Y
        :return: derivates of the output layer
        '''
        self.derivates.append(self.output - Y)

        derivate_Z = self.derivates[0] * softmax_derivate(self.input)
        self.derivates.append(derivate_Z)

        derivate_W = self.derivates[1] * Y
        self.derivates.append(derivate_W)

    def hidden_derivates(self, prev_output, next_layer):
        '''
        :param prev_output: matrix
        :param next_layer: next layer, class instance
        :return: derivates of current layer, except the output layer
        '''
        self.derivates.append(next_layer.weights.T.dot(next_layer.derivates[1]))

        if self.activation =='tanh':
            derivate_Z = self.derivates[0] * tanh_derivate(self.input)
        else:
            derivate_Z = self.derivates[0] * relu_derivate(self.input)
        self.derivates.append(derivate_Z)

        # add regularization
        if self.reg =='l2':
            derivate_W = self.derivates[1] * prev_output + self.lambd / self.input.shape[1] * self.weights
        else:
            derivate_W = (1 + self.lambd / self.input.shape[1]) * self.derivates[1] * prev_output
        self.derivates.append(derivate_W)

    def update_parameters(self, alpha):
        '''
        :param alpha: learning rate
        :return: learned parameters, include: weights, bias
        '''
        self.weights -= alpha * self.derivates[2]
        self.bias -= alpha * self.derivates[1]


class MomentumNetwork(NetworkLayers):

    def __init__(self, beta=0.9, **kwargs):
        '''
        :param beta: used to update momentum, float
        :param kwargs: (layer, neurons, activation, reg, lambd, drop, final)
        '''
        super(MomentumNetwork, self).__init__(**kwargs)
        self.beta = beta
        self.momentums = [np.zeros(self.weights.shape), np.zeros(self.bias.shape)]

    def momentum(self):
        self.momentums[0] = self.beta * self.momentums[0] + self.beta * self.derivates[2]
        self.momentums[1] = self.beta * self.momentums[1] + self.beta * self.derivates[1]

    def update_parameters(self, alpha):
        self.weights -= alpha * self.momentums[0]
        self.bias -= alpha * self.momentums[1]


class RMSNetwork(NetworkLayers):

    def __init__(self, beta2=0.999, epsilon=1e-8, **kwargs):
        '''
        :param beta2: used to update RMS
        :param epsilon: avoid divided by zero
        :param kwargs: (layer, neurons, activation, reg, lambd, drop, final)
        '''
        super(NetworkLayers, self).__init__(**kwargs)
        self.beta2 = beta2
        self.epsilon = epsilon
        self.quadric_momentums = [np.zeros(self.weights.shape), np.zeros(self.bias.shape)]

    def quadric_momentum(self):
        self.quadric_momentums[0] = self.beta2 * self.quadric_momentums[0] + (1 - self.beta2) * self.derivates[2] ** 2
        self.quadric_momentums[1] = self.beta2 * self.quadric_momentums[1] + (1 - self.beta2) * self.derivates[1] ** 2

    def update_parameters(self, alpha):
        self.weights -= alpha * self.derivates[2] / np.sqrt(self.quadric_momentums[0] + self.epsilon)
        self.bias -= alpha * self.derivates[1] / np.sqrt(self.quadric_momentums[1] + self.epsilon)

class AdamNetwork(MomentumNetwork, RMSNetwork):

    def __init__(self, **kwargs):
        '''
        :param kwargs: (layer, neurons, activation, final, reg, lambd, drop, beta, beta2, epsilon)
        '''
        super(AdamNetwork, self).__init__(**kwargs)

    def update_parameters(self, alpha):
        self.weights -= alpha * self.momentums[0] / np.sqrt(self.quadric_momentums[0] + self.epsilon)
        self.bias -= alpha * self.momentums[1] / np.sqrt(self.quadric_momentums[1] + self.epsilon)

