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

from torch.autograd import Variable
import matplotlib.pyplot as plt

from neural_nets import test_nn
from sigma_input import input_to_onehot
from sigma_net import SigmaNet
from validations_ids import get_validation_ids
import os

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '2'
LEARNING_RATE_DEFAULT = 3e-3
MAX_STEPS_DEFAULT = 3000000
BATCH_SIZE_DEFAULT = 8
EVAL_FREQ_DEFAULT = 1


FLAGS = None


def accuracy(predictions, targets, x):
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

    train = torch.narrow(x, 1, 25, 9)

    mean = torch.mean(train, dim=1)
    sigma = torch.std(train, dim=1)

    result = mean + torch.mul(torch.transpose(predictions, 0, 1), sigma)

   # result = result.view(BATCH_SIZE_DEFAULT, 1)

    result = result.detach().numpy()

    preds = np.round(result)
    targets = np.round(targets)

    res = preds == targets

    sum = np.sum(res)

    accuracy = sum / float(targets.shape[0])

    return accuracy


def train():
    """
    Performs training and evaluation of MLP model.
    TODO:
    Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """
    # Set the random seeds for reproducibility
    # np.random.seed(42)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("cuda")
    else:
        device = torch.device('cpu')

    script_directory = os.path.split(os.path.abspath(__file__))[0]
    filepath = 'grubbyStarSigma.model'
    model_to_train = os.path.join(script_directory, filepath)  # EXCEPT CROSS ENTROPY!

    validation_games = 50

    onehot_input, y,_ = input_to_onehot('new_predictions')

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

    model = SigmaNet(onehot_input.shape[1])
    print(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE_DEFAULT, momentum=0.9, weight_decay=1e-5)

    accuracies = []
    losses = []
    vag_losses = []
    max_acc = 0
    min_loss = 1000

    vag_games = get_validation_ids()
    vag_games = np.array(vag_games)
    vag_ids = vag_games[-200:]
    vag_input = onehot_input[vag_ids, :]
    vag_targets = y[vag_ids]

    for epoch in range(1):
        val_ids = [i for i in range(onehot_input.shape[0] - validation_games, onehot_input.shape[0])]
        val_ids = np.append(val_ids, vag_ids)
        val_ids = np.unique(val_ids)
        val_ids = np.array(val_ids)
        print(len(val_ids), "val ids")
        print(val_ids)

        train_ids = [i for i in range(onehot_input.shape[0]) if i not in val_ids]

        X_train = onehot_input[train_ids, :]
        y_train = y[train_ids]

        X_test = onehot_input[val_ids, :]
        y_test = y[val_ids]

        print("epoch " + str(epoch))

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
            loss = center_my_loss(output, y_train_batch, X_train_batch)

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

                acc = accuracy(pred, targets, x)
                targets = np.reshape(targets, (len(X_test), -1))
                targets = Variable(torch.FloatTensor(targets))

                calc_loss = center_my_loss(pred, targets, x)

                accuracies.append(acc)
                losses.append(calc_loss.item())

                ###################

                ids = np.array(range(len(X_train)))
                x = X_train[ids, :]
                targets = y_train[ids]

                x = np.reshape(x, (len(X_train), -1))

                x = Variable(torch.FloatTensor(x))

                pred = model.forward(x)
                train_acc = accuracy(pred, targets, x)

                targets = np.reshape(targets, (len(X_train), -1))

                targets = Variable(torch.FloatTensor(targets))

                train_loss = center_my_loss(pred, targets, x)

                ########## VAG #############

                BATCH_SIZE_DEFAULT = len(vag_ids)
                ids = np.array(range(BATCH_SIZE_DEFAULT))
                x = vag_input
                targets = vag_targets

                x = np.reshape(x, (BATCH_SIZE_DEFAULT, -1))
                x = Variable(torch.FloatTensor(x))

                pred = model.forward(x)
                vag_acc = accuracy(pred, targets, x)

                targets = np.reshape(targets, (BATCH_SIZE_DEFAULT, -1))
                targets = Variable(torch.FloatTensor(targets))

                vag_tensor = np.reshape(vag_input, (BATCH_SIZE_DEFAULT, -1))
                vag_tensor = Variable(torch.FloatTensor(vag_tensor))
                vag_loss = center_my_loss(pred, targets, vag_tensor)
                vag_losses.append(vag_loss.item())

                p = 1
                if min_loss > (p * calc_loss.item() + (1-p) * train_loss.item()):
                    min_loss = (p * calc_loss.item() + (1-p) * train_loss.item())
                    torch.save(model, model_to_train)

                    print("iteration: " + str(iteration) + " train acc " + str(train_acc) + " val acc " + str(
                        acc) + " train loss " + str(train_loss.item()) + " val loss " + str(
                        calc_loss.item()) + " vag acc: " + str(vag_acc) + " vag loss: " + str(vag_loss.item()))

    #torch.save(model, model_to_train)
    test_nn.test_all(model_to_train)
    print(model_to_train)
    print("maxx acc")
    print(max_acc)
    plt.plot(accuracies)
    plt.ylabel('accuracies')
    plt.show()

    plt.plot(vag_losses, 'r')
    plt.plot(losses, 'b')
    plt.ylabel('losses')
    plt.show()


def center_my_loss(output, target, train):
    train = torch.narrow(train, 1, 25, 9)

    mean = torch.mean(train, dim=1)
    sigma = torch.std(train, dim=1)

    result = mean + torch.mul(2 * torch.transpose(output, 0, 1), sigma)

    result = result.view(target.shape[0], 1)

    log = torch.log(1 - torch.abs(result - target))

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