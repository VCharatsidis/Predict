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
import copy
import test_nn

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '2'
LEARNING_RATE_DEFAULT = 2e-5
MAX_STEPS_DEFAULT = 200000
BATCH_SIZE_DEFAULT = 32
EVAL_FREQ_DEFAULT = 1


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
    predictions = predictions.flatten()
    preds = np.round(predictions)
    # print(preds)
    # print(targets)
    result = preds == targets

    sum = np.sum(result)

    accuracy = sum / float(targets.shape[0])

    return accuracy


def get_input(enable_formula = False):
    f = open("Grubb.txt", "r")

    contents = f.readlines()

    data = []
    counter = 0
    for l in contents:
        X = l.split('-')

        X[0] = float(X[0])
        X[4] = float(X[4])
        X[5] = float(X[5])

        if X[5] > 200:
            X[5] = 200

        X[6] = X[6].rstrip("\n")

        if X[4] < 58:
            continue

        X = np.array(X)
        data.append(X)

    return data


def center(X):

    newX = X - np.mean(X, axis=0)
    return newX


def standardize(X):

    newX = center(X) / np.std(X, axis=0)

    return newX


def input_to_onehot():
    labelencoder = LabelEncoder()
    input = get_input()
    input = np.array(input)

    y = input[:, 0]
    y = [float(x) for x in y]
    y = np.array(y)

    input = input[:, 1:]

    input[:, 0] = labelencoder.fit_transform(input[:, 0])
    input[:, 0] = [float(x) for x in input[:, 0]]

    input[:, 1] = labelencoder.fit_transform(input[:, 1])
    input[:, 1] = [float(x) for x in input[:, 1]]

    input[:, 2] = labelencoder.fit_transform(input[:, 2])
    input[:, 2] = [float(x) for x in input[:, 2]]

    input[:, 5] = labelencoder.fit_transform(input[:, 5])
    input[:, 5] = [float(x) for x in input[:, 5]]

    onehotencoder = OneHotEncoder(categorical_features=[0, 1, 2, 5])
    onehot_input = onehotencoder.fit_transform(input).toarray()

    not_standardized_input = copy.deepcopy(onehot_input)

    print("standardize")
    onehot_input[:, -1] = standardize(onehot_input[:, -1])
    onehot_input[:, -2] = standardize(onehot_input[:, -2])

    return onehot_input, y, not_standardized_input

def train():
    """
    Performs training and evaluation of MLP model.
    TODO:
    Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """
    # Set the random seeds for reproducibility
    np.random.seed(42)

    model_to_train = 'grubbyStarTest.model'

    validation_games = 0

    onehot_input, y, _ = input_to_onehot()

    val_ids = np.random.choice(onehot_input.shape[0], size=validation_games, replace=False)
    train_ids = [i for i in range(onehot_input.shape[0]) if i not in val_ids]

    X_train = onehot_input[train_ids, :]
    y_train = y[train_ids]

    # X_train = onehot_input[0: -validation_games, :]
    # y_train = y[0: -validation_games]

    print("X train")

    print(X_train.shape)
    print(y_train.shape)

    X_test = onehot_input[val_ids, :]
    y_test = y[val_ids]

    # X_test = onehot_input[-validation_games:, :]
    # y_test = y[-validation_games:]

    print("X test")

    print(X_test.shape)
    print(y_test.shape)

    print(onehot_input.shape)
    print(onehot_input.shape[1])

    model = MLP(onehot_input.shape[1])
    print(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE_DEFAULT, momentum=0.9, weight_decay=1e-5)

    accuracies = []
    losses = []
    max_acc = 0
    for iteration in range(MAX_STEPS_DEFAULT):
        BATCH_SIZE_DEFAULT = 32
        model.train()

        ids = np.random.choice(X_train.shape[0], size=BATCH_SIZE_DEFAULT, replace=False)

        X_train_batch = X_train[ids, :]
        y_train_batch = y_train[ids]

        X_train_batch = np.reshape(X_train_batch, (BATCH_SIZE_DEFAULT, -1))
        X_train_batch = Variable(torch.FloatTensor(X_train_batch))

        output = model.forward(X_train_batch)

        y_train_batch = np.reshape(y_train_batch, (BATCH_SIZE_DEFAULT, -1))
        y_train_batch = Variable(torch.FloatTensor(y_train_batch))

        loss = nn.functional.binary_cross_entropy(output, y_train_batch)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % EVAL_FREQ_DEFAULT == 0:
            model.eval()


            BATCH_SIZE_DEFAULT = len(X_train)
            ids = np.array(range(BATCH_SIZE_DEFAULT))
            x = X_train[ids, :]
            targets = y_train[ids]

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
                torch.save(model, model_to_train)
                print("iteration: " + str(iteration) + " total accuracy " + str(acc) + " total loss " + str(
                    calc_loss.item()))
                #break;

    test_nn.test_all(model_to_train)
    print(model_to_train)
    print("maxx acc")
    print(max_acc)
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