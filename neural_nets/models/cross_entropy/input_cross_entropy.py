import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import copy
import os


def get_predictions():
    script_directory = os.path.split(os.path.abspath(__file__))[0]
    filepath = 'C:\\Users\\chara\\PycharmProjects\\PredictBet\\logs\\Grubb.txt'
    grubb = os.path.join(script_directory, filepath)

    f = open(grubb, "r")

    contents = f.readlines()
    data = []

    for line in contents:
        X = line.split('-')
        input_x = []

        X[0] = float(X[0])
        X[4] = float(X[4])
        X[5] = float(X[5])

        X[6] = X[6].rstrip("\n")

        input_x.append(X[0])
        input_x.append(X[1])
        input_x.append(X[2])
        input_x.append(X[3])
        input_x.append(X[4])
        input_x.append(X[5])
        input_x.append(X[6])

        data.append(input_x)

    return data


def center(X):
    newX = X - np.mean(X, axis=0)
    return newX


def standardize(X):
    newX = center(X) / np.std(X, axis=0)
    return newX


def cross_entropy_input_to_onehot():
    labelencoder = LabelEncoder()
    input = get_predictions()

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

    onehot_input[:, -1] = standardize(onehot_input[:, -1])
    onehot_input[:, -2] = standardize(onehot_input[:, -2])

    return onehot_input, y, not_standardized_input
