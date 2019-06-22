import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

f = open("../logs/Grubb.txt", "r")
contents = f.readlines()


def get_input():
    input = []
    for l in contents:

        X = l.split('-')

        X[4] = int(X[4])
        if X[4] < 55:
            continue

        X[5] = int(X[5])

        if X[5] < 15:
            X[4] = int(X[4] * 0.88)
        elif X[5] < 30:
            X[4] = int(X[4] * 0.93)

        X[6] = X[6].rstrip("\n")

        X = np.array(X)
        input.append(X)

    input = np.array(input)
    labelencoder = LabelEncoder()

    y = input[:, 0]
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

    return onehot_input, input, y
