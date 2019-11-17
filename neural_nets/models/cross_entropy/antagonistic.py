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
LEARNING_RATE_DEFAULT = 1e-3
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

    LEARNING_RATE_DEFAULT = 3e-3
    MAX_STEPS_DEFAULT = 4000000

    cnet_a = CrossNet2(onehot_input.shape[1])
    script_directory = os.path.split(os.path.abspath(__file__))[0]
    filepath = 'grubbyStarCE2.model'
    model_to_train = os.path.join(script_directory, filepath)

    cnet_b = SimpleMLP(onehot_input.shape[1])
    script_directory_b = os.path.split(os.path.abspath(__file__))[0]
    filepath_b = 'grubbyStarCrossEntropy.model'
    model_b = os.path.join(script_directory_b, filepath_b)

    print(cnet_a)
    print(onehot_input.shape)
    print(onehot_input.shape[1])

    optimizer = torch.optim.SGD(cnet_a.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-5)
    optimizer_b = torch.optim.SGD(cnet_b.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-5)

    accuracies = []
    losses = []
    vag_losses = []
    min_loss = 100
    min_loss_b = 100

    vag_games = get_validation_ids()
    vag_games = np.array(vag_games)
    vag_ids = vag_games[-200:]
    validation_games = 100
    vag_input = onehot_input[vag_ids, :]
    vag_targets = y[vag_ids]

    for epoch in range(1):
        val_ids = [i for i in range(onehot_input.shape[0]-validation_games, onehot_input.shape[0])]
        val_ids = np.append(val_ids, vag_ids)
        val_ids = np.unique(val_ids)
        val_ids = np.array(val_ids)
        print(len(val_ids), "val ids")
        print(val_ids)

        train_ids = [i for i in range(onehot_input.shape[0]) if i not in val_ids]

        X_train = onehot_input[train_ids, :]
        print(X_train.shape)
        y_train = y[train_ids]

        X_test = onehot_input[val_ids, :]
        y_test = y[val_ids]

        print("epoch " + str(epoch))
        saturation = 1
        p = 1
        bce = True
        ace = True

        for iteration in range(MAX_STEPS_DEFAULT):
            BATCH_SIZE_DEFAULT = 8
            cnet_a.train()
            cnet_b.train()
            if iteration % 20000 == 0:
                # saturation *= 0.5
                # saturation = max(0.5, saturation)
                print(iteration)
                print(saturation)

            ids = np.random.choice(X_train.shape[0], size=BATCH_SIZE_DEFAULT, replace=False)

            X_train_batch = X_train[ids, :]
            y_train_batch = y_train[ids]

            X_train_batch = np.reshape(X_train_batch, (BATCH_SIZE_DEFAULT, -1))
            X_train_batch = Variable(torch.FloatTensor(X_train_batch))

            output = cnet_a.forward(X_train_batch)
            output_b = cnet_b.forward(X_train_batch)

            y_train_batch = np.reshape(y_train_batch, (BATCH_SIZE_DEFAULT, -1))
            y_train_batch = Variable(torch.FloatTensor(y_train_batch))

            if iteration % 1 == 0:
                loss = center_my_loss(output, y_train_batch, output_b, saturation)
            else:
                loss = torch.nn.functional.binary_cross_entropy(output, y_train_batch)

            if True:
                loss_b = center_my_loss(output_b, y_train_batch, output, saturation)
            else:
                loss_b = torch.nn.functional.binary_cross_entropy(output_b, y_train_batch)

            ce_loss = torch.nn.functional.binary_cross_entropy(output, y_train_batch)
            ce_loss_b = torch.nn.functional.binary_cross_entropy(output_b, y_train_batch)

            if iteration % EVAL_FREQ_DEFAULT == 0:
                cnet_a.eval()
                cnet_b.eval()

                ids = np.array(range(len(X_test)))

                x = X_test[ids, :]
                targets = y_test[ids]

                x = np.reshape(x, (len(X_test), -1))

                x = Variable(torch.FloatTensor(x))

                pred = cnet_a.forward(x)
                pred_b = cnet_b.forward(x)
                acc = accuracy(pred, targets)

                targets = np.reshape(targets, (len(X_test), -1))
                targets = Variable(torch.FloatTensor(targets))

                calc_loss = torch.nn.functional.binary_cross_entropy(pred, targets)
                calc_loss_b = torch.nn.functional.binary_cross_entropy(pred_b, targets)

                accuracies.append(acc)

                ###################

                if p*calc_loss.item()+(1-p)*ce_loss.item() < p*calc_loss_b.item()+(1-p)*ce_loss_b.item():
                    cnet_b.train()
                    cnet_b.zero_grad()
                    loss_b.backward(retain_graph=True)
                    optimizer_b.step()
                    cnet_b.eval()

                    if min_loss_b > calc_loss_b.item():
                        min_loss_b = calc_loss_b.item()
                        torch.save(cnet_b, model_b)

                        ids = np.array(range(len(X_train)))
                        x = X_train[ids, :]
                        targets = y_train[ids]

                        x = np.reshape(x, (len(X_train), -1))

                        x = Variable(torch.FloatTensor(x))

                        pred_b = cnet_b.forward(x)

                        train_acc = accuracy(pred_b, targets)

                        targets = np.reshape(targets, (len(X_train), -1))
                        targets = Variable(torch.FloatTensor(targets))

                        train_loss = torch.nn.functional.binary_cross_entropy(pred_b, targets)
                        losses.append(train_loss.item())

                        print("iteration: " + str(iteration) + " train acc " + str(train_acc) + " val acc " + str(
                            acc) + " a " + str(round(calc_loss.item()*1000)/1000) + " b " + str(
                            round(calc_loss_b.item() * 1000)/1000))

                if p*calc_loss.item()+(1-p)*ce_loss.item() > p*calc_loss_b.item()+(1-p)*ce_loss_b.item():
                    cnet_a.train()
                    cnet_a.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    cnet_a.eval()

                    if min_loss > calc_loss.item():
                        min_loss = calc_loss.item()
                        torch.save(cnet_a, model_to_train)

                        ids = np.array(range(len(X_train)))
                        x = X_train[ids, :]
                        targets = y_train[ids]

                        x = np.reshape(x, (len(X_train), -1))

                        x = Variable(torch.FloatTensor(x))

                        pred = cnet_a.forward(x)

                        train_acc = accuracy(pred, targets)

                        targets = np.reshape(targets, (len(X_train), -1))
                        targets = Variable(torch.FloatTensor(targets))

                        print("iteration: " + str(iteration) + " train acc " + str(train_acc) + " val acc " + str(
                            acc) + " a " + str(round(calc_loss.item()*1000)/1000) + " b " + str(
                            round(calc_loss_b.item() * 1000)/1000))

    test_nn.test_all(model_to_train)
    print(model_to_train)

    plt.plot(accuracies)
    plt.ylabel('accuracies')
    plt.show()

    plt.plot(vag_losses, 'r')
    plt.plot(losses, 'b')
    plt.ylabel('losses')
    plt.show()

import copy

def center_my_loss(output, target, output_b, saturation):
    copy_b = output_b.detach().numpy()
    copy_b = copy.deepcopy(copy_b)
    copy_b = torch.FloatTensor(copy_b)

    cross_entropy_loss = torch.nn.functional.binary_cross_entropy(output, target)
    unit_bet = 0.2

    hero_won = torch.ceil(target - 0.5)
    hero_loss = 1 - hero_won

    mean_bet = (output + copy_b) / 2
    proportion_bet = (1 - mean_bet) / mean_bet
    opposite_proportion = mean_bet / (1-mean_bet)

    means_greater_than_half = torch.ceil(mean_bet - 0.5)
    mean_smaller_half = 1 - means_greater_than_half

    max = torch.max(output, copy_b)
    max_is_a = torch.sign(max - copy_b)
    max_is_b = 1 - max_is_a

    a = max_is_a * hero_won * means_greater_than_half * unit_bet
    b = max_is_a * hero_won * mean_smaller_half * proportion_bet * unit_bet

    c = max_is_a * hero_loss * means_greater_than_half * (-opposite_proportion * unit_bet)
    d = max_is_a * hero_loss * mean_smaller_half * (-unit_bet)

    e = max_is_b * hero_won * means_greater_than_half * (-unit_bet)
    f = max_is_b * hero_won * mean_smaller_half * (-proportion_bet * unit_bet)

    g = max_is_b * hero_loss * means_greater_than_half * opposite_proportion * unit_bet
    h = max_is_b * hero_loss * mean_smaller_half * unit_bet

    result = a + b + c + d + e + f + g + h

    result_mean = torch.mean(result)
    result_mean /= saturation

    # max_c = torch.max(cross_entropy_loss, torch.abs(result_mean))
    # max_is_result_mean = torch.abs(torch.sign(max_c - result_mean))
    # loss = - result_mean * max_is_result_mean + ()

    max_c = torch.max(cross_entropy_loss-result_mean, torch.zeros(result_mean.shape))
    loss_is_negative = torch.abs(torch.sign(max_c))
    loss = cross_entropy_loss - result_mean * loss_is_negative
    # print(cross_entropy_loss)
    # print(max_is_result_mean)
    # print("loss")
    #
    # print(torch.abs(loss))
    # print(target)
    # print(output)
    # print(output_b)
    #
    # input()

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