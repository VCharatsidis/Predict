
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def parse_one_hot(xin):
    print(xin)
    data = ""
    if xin[0] ==1:
        data += "Hum "
    elif xin[1]==1:
        data += " Ne "
    elif xin[2]==1:
        data += " Orc "
    elif xin[3]==1:
        data += " Ra "
    elif xin[4]==1:
        data += " Ud"

    data += " vs "

    if xin[7]==1:
        data += " Hum "
    elif xin[8]==1:
        data += " Ne "
    elif xin[9]==1:
        data += " Orc "
    elif xin[10]==1:
        data += " Ra "
    elif xin[11]==1:
        data += " Ud"

    # amazonia = 0
    # concealed = 1
    # echo = 2
    # northren = 3
    # refuge = 4
    # swamped = 5
    # terenas = 6
    # turtle = 7
    # twisted = 8
    print("len xin "+str(len(xin)))
    if len(xin) > 11:
        if xin[12] == 1:
            data += " amazonia "
        elif xin[13] == 1:
            data += " concealed "
        elif xin[14] == 1:
            data += " echo "
        elif xin[15] == 1:
            data += " northren "
        elif xin[16] == 1:
            data += " refuge"
        elif xin[17] == 1:
            data += " swamped "
        elif xin[18] == 1:
            data += " terenas "
        elif xin[19] == 1:
            data += " turtle "
        elif xin[20] == 1:
            data += " twisted"

        if xin[5] == 0:
            data += " tryhard "
        else:
            data += " request"

        data += " "+str(xin[21])+"%"

        data += " se "+str(xin[22])+" games"

    else:
        if xin[10] == 1:
            data += " tryhard "
        else:
            data += " request"

        data += " " + str(xin[11]) + "%"

        data += " se " + str(xin[12]) + " games"

    return data

def parse_x(xin):
    data = "1-"

    if xin[0] == 0:
        data += "Hum-"
    elif xin[0] == 1:
        data += "Ne-"
    elif xin[0] == 2:
        data += "Orc-"
    elif xin[0] == 3:
        data += "Ra-"
    elif xin[0] == 4:
        data += "Ud-"

    if xin[1] == 1:
        data += "t-"
    else:
        data += "r-"

    if xin[2] == 0:
        data += "Hum-"
    elif xin[2] == 1:
        data += "Ne-"
    elif xin[2] == 2:
        data += "Orc-"
    elif xin[2] == 3:
        data += "Ra-"
    elif xin[2] == 4:
        data += "Ud-"

    data += str(xin[3])
    data += "-"+str(xin[4])+"-"

    # amazonia = 0
    # concealed = 1
    # echo = 2
    # northren = 3
    # refuge = 4
    # swamped = 5
    # terenas = 6
    # turtle = 7
    # twisted = 8
    if(len(xin)>5):
        if xin[5] == 0:
            data += "amazonia"
        elif xin[5] == 1:
            data += "concealed"
        elif xin[5] == 2:
            data += "echo"
        elif xin[5] == 3:
            data += "northren"
        elif xin[5] == 4:
            data += "refuge"
        elif xin[5] == 5:
            data += "swamped"
        elif xin[5] == 6:
            data += "terenas"
        elif xin[5] == 7:
            data += "turtle"
        elif xin[5] == 8:
            data += "twisted"
        elif xin[5] == 9:
            data += "ancient"

    return data


################################## Logistic  -------------------------------------------------


def logistic_reg(xin):
    f = open("Grubb.txt", "r")

    contents = f.readlines()

    input = []
    counter = 0
    for l in contents:
        X = l.split('-')

        winrate = int(X[4]) / 100
        games = int(X[5])

        wins = round(winrate * games)
        losses = round((1 - winrate) * games)

        X[4] = wins
        X[5] = losses
        X[6] = X[6].rstrip("\n")

        X = np.array(X)
        input.append(X)

        print(X)
        print(l)
        counter += 1

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

    print(onehot_input[-1])
    print("Logistic regression train")
    clf = LogisticRegression(solver='lbfgs', max_iter=500).fit(onehot_input, y)


    winrate = xin[3] / 100
    games = xin[4]
    wins = round(winrate * games)
    losses = games - wins

    xin = [xin[0], xin[1], xin[2], wins, losses, xin[5]]

    onehot_encoded = []

    letter = [0 for _ in range(5)]
    letter[xin[0]] = 1
    for i in letter:
        onehot_encoded.append(i)

    letter = [0 for _ in range(2)]
    letter[xin[1]] = 1
    for i in letter:
        onehot_encoded.append(i)

    letter = [0 for _ in range(5)]
    letter[xin[2]] = 1
    for i in letter:
        onehot_encoded.append(i)

    letter = [0 for _ in range(10)]
    letter[xin[5]] = 1
    for i in letter:
        onehot_encoded.append(i)

    onehot_encoded = np.array(onehot_encoded)
    onehot_encoded.flatten()

    onehot_encoded = np.append(onehot_encoded, xin[3])
    onehot_encoded = np.append(onehot_encoded, xin[4])

    onehot_encoded.flatten()

    print("manual one hot " + str(onehot_encoded))
    print(len(onehot_encoded))

    print("Logistic regression predict")
    y_pred_logistic = clf.predict_proba([onehot_encoded])
    print(y_pred_logistic)

    return y_pred_logistic


xin = [2, 1, 4, 86, 29, 2]
logistic_reg(xin)









