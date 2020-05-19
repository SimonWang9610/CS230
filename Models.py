import numpy as np

from NeuralNetwork import NetworkLayers

class Models(object):

    def __init__(self, batch, optimizer=0):
        self.batch = batch
        self.optimizer = optimizer


def forward_propagation(networks, data):
    for i in range(len(networks)):
        if networks[i].weights is None:
            if i == 0:
                networks[i].initialize_weights_bias(data.shape[0])
                networks[i].calculate_input_output(data)
            else:
                networks[i].initialize_weights_bias(networks[i-1].neurons)
                networks[i].calculate_input_output(networks[i-1].output)

def compute_cost(output, labels, log_loss=False):
    if log_loss:
        cost = - 1 / labels.shape[1] * np.sum(labels * np.log(output) + (1 - labels) * np.log(1 - output))
    else:
        cost = - 1 / labels.shape[1] * np.sum((labels - output) ** 2)

    return cost

def backward_propagation(networks, data, labels):
    for i in reversed(range(len(networks))):
        if i == len(networks):
            networks[i].final_derivates(labels)
        elif i == 0:
            networks[i].hidden_derivates(data, networks[i+1])
        else:
            networks[i].hidden_derivates(networks[i-1].output, networks[i+1])

def update_optimization_parameters(networks, alpha):
    pass

def update_parameters(networks, alpha):
    for i in range(len(networks)):
        networks[i].update_parameters(alpha)
network_setting = []

def momentum(networks):
    for i in range(len(networks)):
        networks[i].momentum()

def rms_momentum(networks):
    for i in range(len(networks)):
        networks[i].quadric_momentum()

def evaluate(networks, data,labels):
    pass

def count_correct(output, labels):
    pass

def build_model(X_train, y_train, X_test, y_test, iterations, networks, alpha, batch_size, model):

    for epoch in range(iterations):
        cost = 0
        correct = 0
        batch_num = X_train.shape[1] / batch_size
        for i in range(batch_num):
            data = X_train[i*batch_size, (i+1)*batch_num]
            labels = y_train[i*batch_size, (i+1)*batch_num]

            forward_propagation(networks, data)
            correct += count_correct(networks[-1].output, labels)

            loss = compute_cost(networks[-1].output, labels)
            cost += loss

            backward_propagation(networks, data, labels)

            # calculate parameters according to different algorithms
            if model == 'Momentum':
                momentum(networks)
            elif model == 'RMS':
                rms_momentum(networks)
            elif model == 'Adam':
                momentum(networks)
                rms_momentum(networks)

            update_parameters(networks, alpha)
        train_accuracy = correct / X_train.shape[1]
        test_accuracy = evaluate(networks, X_test, y_test)

        if epoch % 10 == 0:
            print(f'Cost for {epoch}-th: {cost}, Train-Accuracy:{train_accuracy}, Test-Accuracy:{test_accuracy}')
