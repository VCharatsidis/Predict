import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import copy
import os

maps = []


def get_predictions():
    script_directory = os.path.split(os.path.abspath(__file__))[0]
    filepath = '..\\logs\\'
    targets = os.path.join(script_directory, filepath + 'automagic.txt')
    f = open(targets, "r")

    data = []
    vagelis_games = []

    contents = f.readlines()
    counter = 0
    for line in contents:
        X = line.split('-')
        if int(X[4]) < 55:
            counter += 1
            continue

        if len(X) > 18:
            vag = copy.deepcopy(X[16])
            vag = vag.rstrip("\n")
            vag = vag.replace('%', '')
            vag = int(vag)
            if vag != 0:
                vagelis_games.append(counter)

        processed_X = []
        max_prediction = 3.
        min_prediction = 98.

        for x in X:
            if '%' in x:
                x = x.rstrip("\n")
                x = x.replace('%', '')
                pred = float(x)

                if pred > max_prediction:
                    max_prediction = pred

                if pred > 0:
                    if pred < min_prediction:
                        min_prediction = pred

        max_saturation = 0
        min_saturation = 0

        max_prediction = min(max_prediction - max_saturation, 98.)
        min_prediction = max(min_prediction + min_saturation, 3.)

        if float(X[0]) > 0.5:
            X[0] = float(max_prediction / 100.)
        else:
            X[0] = float(min_prediction / 100.)

        X[4] = float(X[4])
        X[5] = float(X[5])

        X[6] = X[6].rstrip("\n")
        if X[6] not in maps:
            maps.append(X[6])

        processed_X.append(X[0])
        processed_X.append(X[1])
        processed_X.append(X[2])
        processed_X.append(X[3])
        processed_X.append(X[4])
        processed_X.append(X[5])
        processed_X.append(X[6])

        processed_X = np.array(processed_X)
        data.append(processed_X)
        counter += 1

    return data, vagelis_games


def center(X):
    newX = X - np.mean(X, axis=0)
    return newX


def standardize(X):
    newX = center(X) / np.std(X, axis=0)
    return newX


def input_to_onehot():
    labelencoder = LabelEncoder()
    input, _ = get_predictions()

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


def filter(list):
    filter_55 = []
    for i in range(len(list)):
        z = list[i].split("-")
        if int(z[4]) < 55:
            continue
        filter_55.append(list[i])

    return filter_55


def check_input():
    predictions = open("../logs/new_predictions.txt", "r")
    results = open("../logs/automagic.txt", "r")

    contents_pred = predictions.readlines()
    contents_res = results.readlines()

    filtered_results = filter(contents_res)

    for i in range(len(contents_pred)):
        print(i)

        print(contents_pred[i])
        print(filtered_results[i])
        if contents_pred[i][0] != filtered_results[i][0]:
            print("ERROR")
            print(i)
            break


#check_input()