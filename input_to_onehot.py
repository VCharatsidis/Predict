import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import copy

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

    onehot_input[:, -1] = standardize(onehot_input[:, -1])
    onehot_input[:, -2] = standardize(onehot_input[:, -2])

    return onehot_input, y, not_standardized_input