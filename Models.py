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
                networks[i].initialize_parameters(data.shape[0])
                networks[i].calculate_input_output(data)
            else:
                networks[i].initialize_parameters(networks[i-1].neurons)
                networks[i].calculate_input_output(networks[i-1].output)
        else:
            if i == 0:
                networks[i].calculate_input_output(data)
            else:
                networks[i].calculate_input_output(networks[i - 1].output)

def compute_cost(output, labels, log_loss=False):
    if log_loss:
        cost = - 1 / labels.shape[1] * np.sum(labels * np.log(output) + (1 - labels) * np.log(1 - output))
    else:
        cost = 1 / labels.shape[1] * np.sum((labels - output) ** 2)

    return cost

def backward_propagation(networks, data, labels):
    for i in reversed(range(len(networks))):
        if i == len(networks) - 1:
            networks[i].final_derivates(networks[i-1].output, labels)
        elif i == 0:
            networks[i].hidden_derivates(data, networks[i+1])
        else:
            networks[i].hidden_derivates(networks[i-1].output, networks[i+1])

def update_parameters(networks, alpha):
    for i in range(len(networks)):
        networks[i].update_parameters(alpha)


def momentum(networks):
    for i in range(len(networks)):
        networks[i].momentum()

def rms_momentum(networks):
    for i in range(len(networks)):
        networks[i].quadric_momentum()

def count_correct(output, labels):
    count = 0
    for i in range(output.shape[1]):
        if np.argmax(output[:,i:i+1]) == np.argmax(labels[:,i:i+1]):
            count += 1
    return count

def evaluate(networks, data,labels):
    forward_propagation(networks, data)
    count = count_correct(networks[-1].output, labels)
    return count

def scale_labels(Y, targtes):
    temp = np.zeros((targtes, Y.shape[1]))
    for i in range(Y.sahep[1]):
        temp[int(Y[i]),i] = 1
    return temp

def build_model(X_train, y_train, X_test, y_test, iterations, networks, alpha, batch_size, model='gd', log_loss=False):

    for epoch in range(iterations):
        cost = 0
        correct = 0

        batch_num = X_train.shape[1] // batch_size
        rest = X_train.shape[1] % batch_size

        for i in range(batch_num):

            if rest != 0 and i == batch_num - 1:
                data = X_train[:, i*batch_size: ]
                labels = y_train[:, i*batch_size:]
            else:
                data = X_train[:, i * batch_size: (i + 1) * batch_size]
                labels = y_train[:, i * batch_size: (i + 1) * batch_size]

            forward_propagation(networks, data)
            correct += count_correct(networks[-1].output, labels)

            loss = compute_cost(networks[-1].output, labels, log_loss)
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
        test_accuracy = evaluate(networks, X_test, y_test) / y_test.shape[1]
        if epoch % 10 ==0:
            print(f'Cost for {epoch}-th: {cost}, Train-Accuracy:{train_accuracy}, Test-Accuracy:{test_accuracy}')
