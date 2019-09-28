"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import torch
from cross_net2 import CrossNet2
from cross_net3 import CrossNet3
from cross_net4 import CrossNet4
from simple_net import SimpleMLP
from torch.autograd import Variable
import matplotlib.pyplot as plt
from neural_nets import test_nn
from neural_nets.models.cross_entropy.input_cross_entropy import cross_entropy_input_to_onehot
from neural_nets.input_to_onehot import get_predictions
from neural_nets.validations_ids import get_validation_ids
import os

# CROSS ENTROPY NN

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '2'
LEARNING_RATE_DEFAULT = 5e-5
MAX_STEPS_DEFAULT = 300
BATCH_SIZE_DEFAULT = 8
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

    result = preds == targets

    sum = np.sum(result)

    accuracy = sum / float(targets.shape[0])
    accuracy = accuracy * 1000
    accuracy = round(accuracy)/1000

    return accuracy


def train():
    """
    Performs training and evaluation of MLP model.
    """
    # Set the random seeds for reproducibility
    # np.random.seed(42)

    onehot_input, y, _ = cross_entropy_input_to_onehot()

    LEARNING_RATE_DEFAULT = 1e-3
    MAX_STEPS_DEFAULT = 500000

    model = CrossNet2(onehot_input.shape[1])
    script_directory = os.path.split(os.path.abspath(__file__))[0]
    filepath = 'grubbyStarCE2.model'
    model_to_train = os.path.join(script_directory, filepath)

    print(model)
    print(onehot_input.shape)
    print(onehot_input.shape[1])

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE_DEFAULT, momentum=0.9, weight_decay=1e-5)

    accuracies = []
    losses = []
    vag_losses = []
    min_loss = 100

    vag_games = get_validation_ids()
    vag_games = np.array(vag_games)
    vag_ids = vag_games[-200:]
    validation_games = 70
    vag_input = onehot_input[vag_ids, :]
    vag_targets = y[vag_ids]

    for epoch in range(1):
        val_ids = np.random.choice(onehot_input.shape[0], size=validation_games, replace=False)
        val_ids = np.append(val_ids, vag_ids)
        val_ids = np.unique(val_ids)
        val_ids = np.array(val_ids)

        train_ids = [i for i in range(onehot_input.shape[0]) if i not in val_ids]

        X_train = onehot_input[train_ids, :]
        print(X_train.shape)
        y_train = y[train_ids]

        X_test = onehot_input[val_ids, :]
        y_test = y[val_ids]

        print("epoch " + str(epoch))

        for iteration in range(MAX_STEPS_DEFAULT):
            BATCH_SIZE_DEFAULT = 12
            model.train()
            if iteration % 10000 == 0:
                print(iteration)

            ids = np.random.choice(X_train.shape[0], size=BATCH_SIZE_DEFAULT, replace=False)

            X_train_batch = X_train[ids, :]
            y_train_batch = y_train[ids]

            X_train_batch = np.reshape(X_train_batch, (BATCH_SIZE_DEFAULT, -1))
            X_train_batch = Variable(torch.FloatTensor(X_train_batch))

            output = model.forward(X_train_batch)

            y_train_batch = np.reshape(y_train_batch, (BATCH_SIZE_DEFAULT, -1))
            y_train_batch = Variable(torch.FloatTensor(y_train_batch))

            loss = torch.nn.functional.binary_cross_entropy(output, y_train_batch)

            model.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            if iteration % EVAL_FREQ_DEFAULT == 0:
                model.eval()

                ids = np.array(range(len(X_test)))

                x = X_test[ids, :]
                targets = y_test[ids]

                x = np.reshape(x, (len(X_test), -1))

                x = Variable(torch.FloatTensor(x))

                pred = model.forward(x)
                acc = accuracy(pred, targets)

                targets = np.reshape(targets, (len(X_test), -1))
                targets = Variable(torch.FloatTensor(targets))

                calc_loss = torch.nn.functional.binary_cross_entropy(pred, 0.95 * targets)

                accuracies.append(acc)


                ###################

                ids = np.array(range(len(X_train)))
                x = X_train[ids, :]
                targets = y_train[ids]

                x = np.reshape(x, (len(X_train), -1))

                x = Variable(torch.FloatTensor(x))

                pred = model.forward(x)
                train_acc = accuracy(pred, targets)

                targets = np.reshape(targets, (len(X_train), -1))
                targets = Variable(torch.FloatTensor(targets))

                train_loss = torch.nn.functional.binary_cross_entropy(pred, 0.95*targets)
                losses.append(train_loss.item())

                ########## VAG #############

                BATCH_SIZE_DEFAULT = len(vag_ids)
                ids = np.array(range(BATCH_SIZE_DEFAULT))
                x = vag_input
                targets = vag_targets

                x = np.reshape(x, (BATCH_SIZE_DEFAULT, -1))

                x = Variable(torch.FloatTensor(x))

                pred = model.forward(x)
                vag_acc = accuracy(pred, targets)

                targets = np.reshape(targets, (BATCH_SIZE_DEFAULT, -1))
                targets = Variable(torch.FloatTensor(targets))

                vag_loss = torch.nn.functional.binary_cross_entropy(pred, 0.95*targets)
                vag_losses.append(vag_loss.item())

                p = 0.9
                if min_loss > (p * calc_loss.item() + (1 - p) * train_loss.item()):
                    min_loss = (p * calc_loss.item() + (1 - p) * train_loss.item())
                    torch.save(model, model_to_train)

                    print("iteration: " + str(iteration) + " train acc " + str(train_acc) + " val acc " + str(
                        acc) + " train loss " + str(round(train_loss.item()*1000)/1000) + " val loss " + str(
                        round(calc_loss.item() * 1000)/1000) + " vag acc: " + str(vag_acc) + " vag loss: " + str(round(vag_loss.item()*1000)/1000))


    test_nn.test_all(model_to_train)
    print(model_to_train)

    plt.plot(accuracies)
    plt.ylabel('accuracies')
    plt.show()

    plt.plot(vag_losses, 'r')
    plt.plot(losses, 'b')
    plt.ylabel('losses')
    plt.show()


def center_my_loss(output, target):
    log = target * torch.log(output+0.1) + (1-target) * torch.log(1.05 - output)
    loss = torch.mean(-log)

    return loss

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