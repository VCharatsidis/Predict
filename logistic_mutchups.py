hum_dict = {'Hum': 0, 'Ne': 1, 'Orc': 2, 'Ra': 3, 'Ud': 4}
ne_dict = {'Hum': 5, 'Ne': 6, 'Orc': 7, 'Ra': 8, 'Ud': 9}
orc_dict = {'Hum': 10, 'Ne': 11, 'Orc': 12, 'Ra': 13, 'Ud': 14}
ra_dict = {'Hum': 15, 'Ne': 16, 'Orc': 17, 'Ra': 18, 'Ud': 19}
ud_dict = {'Hum': 20, 'Ne': 21, 'Orc': 22, 'Ra': 23, 'Ud': 24}

race_dict = {0: 'Hum', 1: 'Ne', 2: 'Orc', 3: 'Ra', 4: 'Ud'}

muchups_dicts = {'Hum': hum_dict, 'Ne': ne_dict, 'Orc': orc_dict, 'Ra': ra_dict, 'Ud': ud_dict}


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

################################## Logistic  -------------------------------------------------


def logistic_reg(xin):
    f = open("Grubb.txt", "r")

    contents = f.readlines()

    input = []
    counter = 0
    for l in contents:
        X = l.split('-')

        Grubby_race = X[1]
        opponent_race = X[3]

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

    input[:, 0] = labelencoder.fit_transform(input[:, 0])
    input[:, 0] = [int(x) for x in input[:, 0]]

    input[:, 1] = labelencoder.fit_transform(input[:, 1])
    input[:, 1] = [int(x) for x in input[:, 1]]

    input[:, 3] = labelencoder.fit_transform(input[:, 3])
    input[:, 3] = [int(x) for x in input[:, 3]]


    onehotencoder = OneHotEncoder(categorical_features=[0, 1, 3])
    onehot_input = onehotencoder.fit_transform(input).toarray()

    print("Logistic regression train")
    clf = LogisticRegression(solver='lbfgs', max_iter=500).fit(onehot_input, y)

    Grubby_race = race_dict[xin[0]]
    opponent_race = race_dict[xin[2]]

    xin = [muchups_dicts[Grubby_race][opponent_race], xin[1], xin[3], xin[5]]


    onehot_encoded = []

    letter = [0 for _ in range(25)]
    letter[xin[0]] = 1
    for i in letter:
        onehot_encoded.append(i)

    letter = [0 for _ in range(2)]
    letter[xin[1]] = 1
    for i in letter:
        onehot_encoded.append(i)

    letter = [0 for _ in range(9)]
    letter[xin[3]] = 1
    for i in letter:
        onehot_encoded.append(i)

    onehot_encoded = np.array(onehot_encoded)
    onehot_encoded.flatten()

    onehot_encoded = np.append(onehot_encoded, xin[2])
    #onehot_encoded = np.append(onehot_encoded, xin[3])

    onehot_encoded.flatten()

    print("manual one hot " + str(onehot_encoded))
    print(len(onehot_encoded))

    print("Logistic regression predict")
    y_pred_logistic = clf.predict_proba([onehot_encoded])
    print(y_pred_logistic)

    return y_pred_logistic


#0-Hum-t-Ne-60-660-northren

xin = [0, 1, 0, 86, 3800, 3]
logistic_reg(xin)








