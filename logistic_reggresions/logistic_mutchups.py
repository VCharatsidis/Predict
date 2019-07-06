
hum_dict = {'Hum': 0, 'Ne': 1, 'Orc': 2, 'Ra': 3, 'Ud': 4}
ne_dict = {'Hum': 5, 'Ne': 6, 'Orc': 7, 'Ra': 8, 'Ud': 9}
orc_dict = {'Hum': 10, 'Ne': 11, 'Orc': 12, 'Ra': 13, 'Ud': 14}
ra_dict = {'Hum': 15, 'Ne': 16, 'Orc': 17, 'Ra': 18, 'Ud': 19}
ud_dict = {'Hum': 20, 'Ne': 21, 'Orc': 22, 'Ra': 23, 'Ud': 24}


muchups_dicts = {'Hum': hum_dict, 'Ne': ne_dict, 'Orc': orc_dict, 'Ra': ra_dict, 'Ud': ud_dict}


import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import config

################################## Logistic  -------------------------------------------------


def logistic_reg(xin, path, i):
    f = open(path, "r")

    contents = f.readlines()

    input = []
    counter = 0
    for l in contents:
        if counter == i:
            counter += 1
            continue

        X = l.split('-')

        Grubby_race = X[1]
        opponent_race = X[3]

        X[4] = int(X[4])
        if X[4] < 55:
            counter += 1
            continue

        X[1] = muchups_dicts[Grubby_race][opponent_race]

        X[3] = int(X[4])
        X[4] = X[6].rstrip("\n")

        X = X[:-2]

        X = np.array(X)

        input.append(X)

        counter += 1

    input = np.array(input)

    labelencoder = LabelEncoder()

    y = input[:, 0]
    input = input[:, 1:]

    input[:, 1] = labelencoder.fit_transform(input[:, 1])
    input[:, 1] = [int(x) for x in input[:, 1]]

    input[:, 3] = labelencoder.fit_transform(input[:, 3])
    input[:, 3] = [int(x) for x in input[:, 3]]

    onehotencoder = OneHotEncoder(categorical_features=[0, 1, 3])
    onehot_input = onehotencoder.fit_transform(input).toarray()

    clf = LogisticRegression(solver='lbfgs', max_iter=1000).fit(onehot_input, y)
    clfCV = LogisticRegressionCV(solver='lbfgs', max_iter=1000, cv=10).fit(onehot_input, y)

    Grubby_race = config.races[int(xin[0])]
    opponent_race = config.races[int(xin[2])]

    xin = [muchups_dicts[Grubby_race][opponent_race], int(xin[1]), int(xin[3]), int(xin[5])]
    print("xin logistic regression mu: " + str(xin))

    onehot_encoded = []

    letter = [0 for _ in range(25)]
    letter[xin[0]] = 1
    for i in letter:
        onehot_encoded.append(i)

    letter = [0 for _ in range(2)]
    letter[xin[1]] = 1
    for i in letter:
        onehot_encoded.append(i)

    letter = [0 for _ in range(config.map_number)]
    letter[xin[3]] = 1
    for i in letter:
        onehot_encoded.append(i)

    onehot_encoded = np.array(onehot_encoded)
    onehot_encoded.flatten()

    onehot_encoded = np.append(onehot_encoded, xin[2])

    onehot_encoded.flatten()

    y_pred_logistic = clf.predict_proba([onehot_encoded])
    y_pred_cv = clfCV.predict_proba([onehot_encoded])

    return y_pred_logistic, y_pred_cv


#0-Hum-t-Ne-60-660-northren

# xin = [4, 1, 4, 86, 3800, 7]
# res = logistic_reg(xin)
# print(res)








