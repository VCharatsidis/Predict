"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import torch
from neural_net import MLP
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '2'
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 10000
BATCH_SIZE_DEFAULT = 8
EVAL_FREQ_DEFAULT = 5


FLAGS = None


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    predictions = predictions.detach().numpy()
    predictions =  predictions.flatten()
    preds = np.round(predictions)
    # print(preds)
    # print(targets)
    result = preds == targets

    sum = np.sum(result)

    accuracy = sum / float(targets.shape[0])
    print(accuracy)

    return accuracy


def get_input(enable_formula = False):
    f = open("Grubb.txt", "r")

    contents = f.readlines()

    data = []
    counter = 0
    for l in contents:
        X = l.split('-')

        X[0] = int(X[0])
        X[4] = int(X[4])
        X[5] = int(X[5])
        X[6] = X[6].rstrip("\n")

        X = np.array(X)
        data.append(X)

    print(counter)
    return data


def center(X):
    newX = X - np.mean(X, axis=0)
    return newX


def standardize(X):
    newX = center(X)/np.std(X, axis=0)

    return newX


def train():
    """
    Performs training and evaluation of MLP model.
    TODO:
    Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """
    labelencoder = LabelEncoder()
    input = get_input()
    input = np.array(input)


    y = input[:, 0]
    y = [int(x) for x in y]
    y = np.array(y)

    input = input[:, 1:]

    input[:, 0] = labelencoder.fit_transform(input[:, 0])
    input[:, 0] = [int(x) for x in input[:, 0]]

    input[:, 1] = labelencoder.fit_transform(input[:, 1])
    input[:, 1] = [int(x) for x in input[:, 1]]

    input[:, 2] = labelencoder.fit_transform(input[:, 2])
    input[:, 2] = [int(x) for x in input[:, 2]]

    input[:, 5] = labelencoder.fit_transform(input[:, 5])
    input[:, 5] = [int(x) for x in input[:, 5]]

    onehotencoder = OneHotEncoder(categorical_features=[0, 1, 2, 5])
    onehot_input = onehotencoder.fit_transform(input).toarray()

    print("standardize")
    onehot_input[:, -1] = standardize(onehot_input[:, -1])
    onehot_input[:, -2] = standardize(onehot_input[:, -2])
    #onehot_input = standardize(onehot_input)
    print(onehot_input)

    # Set the random seeds for reproducibility
    np.random.seed(42)

    validation_games = 50

    X_train = onehot_input[0: -validation_games, :]
    y_train = y[0: -validation_games]

    print("X train")

    print(X_train.shape)
    print(y_train.shape)

    X_test = onehot_input[-validation_games:, :]
    y_test = y[-validation_games:]

    print("X test")

    print(X_test.shape)
    print(y_test.shape)

    print(onehot_input.shape)

    model = MLP(onehot_input.shape[1])
    print(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE_DEFAULT, momentum=0.9)

    accuracies = []
    losses = []
    max_acc = 0
    for iteration in range(MAX_STEPS_DEFAULT):
        BATCH_SIZE_DEFAULT = 8
        model.train()

        ids = np.random.choice(X_train.shape[0], size=BATCH_SIZE_DEFAULT, replace=False)
        X_train_batch = X_train[ids, :]
        y_train_batch = y_train[ids]

        X_train_batch = np.reshape(X_train_batch, (BATCH_SIZE_DEFAULT, -1))
        X_train_batch = Variable(torch.FloatTensor(X_train_batch))

        output = model.forward(X_train_batch)

        y_train_batch = np.reshape(y_train_batch, (BATCH_SIZE_DEFAULT, -1))
        y_train_batch = Variable(torch.FloatTensor(y_train_batch))

        # print(output)
        # print(iteration)
        # print(y_train_batch)

        loss = nn.functional.binary_cross_entropy(output, y_train_batch)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % EVAL_FREQ_DEFAULT == 0:
            model.eval()

            BATCH_SIZE_DEFAULT = len(X_test)
            ids = np.array(range(BATCH_SIZE_DEFAULT))
            x = X_test[ids, :]
            targets = y_test[ids]

            x = np.reshape(x, (BATCH_SIZE_DEFAULT, -1))

            x = Variable(torch.FloatTensor(x))

            pred = model.forward(x)


            acc = accuracy(pred, targets)
            targets = np.reshape(targets, (BATCH_SIZE_DEFAULT, -1))
            targets = Variable(torch.FloatTensor(targets))

            calc_loss = nn.functional.binary_cross_entropy(pred, targets)

            accuracies.append(acc)
            losses.append(calc_loss.item())

            if acc > max_acc:
                max_acc = acc
                torch.save(model, 'grubbyStar.model')
            print("total accuracy " + str(acc) + " total loss " + str(calc_loss.item()))

    plt.plot(accuracies)
    plt.ylabel('accuracies')
    plt.show()

    plt.plot(losses)
    plt.ylabel('losses')
    plt.show()
    ########################
    # END OF YOUR CODE    #
    #######################


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()
    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')

    FLAGS, unparsed = parser.parse_known_args()

    main()