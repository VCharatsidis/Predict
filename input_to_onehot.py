import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import copy


def get_predictions(data):
    f = open("predictions.txt", "r")

    contents = f.readlines()


    for line in contents:
        X = line.split('-')

        if int(X[4]) < 57:
            continue

        processed_X = []
        max_prediction = 0.
        min_prediction = 100.
        for x in X:
            if '%' in x:
                x = x.replace('%', '')
                pred = float(x)

                if pred > max_prediction:
                    max_prediction = pred

                if pred > 0:
                    if pred < min_prediction:
                        min_prediction = pred

        max_prediction = min(max_prediction+1, 100.)
        min_prediction = max(min_prediction-1, 0.)

        if X[0] == 1:
            X[0] = float(max_prediction / 100.)
        else:
            X[0] = float(min_prediction / 100.)

        X[4] = float(X[4])
        X[5] = float(X[5])

        if X[5] > 200:
            X[5] = 200

        X[6] = X[6].rstrip("\n")

        processed_X.append(X[0])
        processed_X.append(X[1])
        processed_X.append(X[2])
        processed_X.append(X[3])
        processed_X.append(X[4])
        processed_X.append(X[5])
        processed_X.append(X[6])

        X = np.array(X)

        processed_X = np.array(processed_X)
        data.append(processed_X)

    return data


def get_input():
    f = open("Grubb.txt", "r")

    contents = f.readlines()

    data = []
    counter = 0
    for l in contents:
        if counter > 163:
           break

        X = l.split('-')

        X[0] = float(X[0])
        X[4] = float(X[4])
        X[5] = float(X[5])

        if X[5] > 200:
            X[5] = 200

        X[6] = X[6].rstrip("\n")

        if X[4] < 57:
            continue

        X = np.array(X)
        data.append(X)
        counter += 1

    get_predictions(data)

    print("data: " + str(len(data)))
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
