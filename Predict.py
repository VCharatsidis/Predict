f = open("Grubb.txt", "r")
import numpy as np
import pandas as pd


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

    if xin[5]==1:
        data += " Hum "
    elif xin[6]==1:
        data += " Ne "
    elif xin[7]==1:
        data += " Orc "
    elif xin[8]==1:
        data += " Ra "
    elif xin[9]==1:
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
    if len(xin) > 13:
        if xin[10] == 1:
            data += " amazonia "
        elif xin[11] == 1:
            data += " concealed "
        elif xin[12] == 1:
            data += " echo "
        elif xin[13] == 1:
            data += " northren "
        elif xin[14] == 1:
            data += " refuge"
        elif xin[15] == 1:
            data += " swamped "
        elif xin[16] == 1:
            data += " terenas "
        elif xin[17] == 1:
            data += " turtle "
        elif xin[18] == 1:
            data += " twisted"

        if xin[19] == 1:
            data += " tryhard "
        else:
            data += " request"

        data += " "+str(xin[20])+"%"

        data += " se "+str(xin[21])+" games"

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

    return data

def less_buckets(games):
    bucket = 0

    if games <= 20:
        bucket = 1
    elif games <= 60:
        bucket = 2
    elif games <= 150:
        bucket = 3
    else:
        bucket = 4

    return bucket

def bucket_games(games):
    bucket = 0

    if games <= 12:
        bucket = 12
    elif games <= 20:
        bucket = 20
    elif games <= 35:
        bucket = 35
    elif games <= 60:
        bucket = 60
    elif games <= 120:
        bucket = 120
    else:
        bucket = 1000

    return bucket

contents = f.readlines()

input = []
counter = 0
for l in contents:

    X = l.split('-')
    # z = int(X[4])
    # stats_adipalou = (round(z/5))*5
    # X[4] = stats_adipalou
    # games_adipalou = int(X[5])
    # b = less_buckets(games_adipalou)
    # X[5] = b
    # X[6] = X[6].rstrip("\n")

    X[4] = int(X[4])
    X[5] = int(X[5])
    X[6] = X[6].rstrip("\n")


    X = np.array(X)
    input.append(X)

    # print("bucket "+str(b))
    # print("stats adipalou rounded "+str(stats_adipalou))
    # print("stats adipalou "+str(z))

    print(X)
    print(l)
    counter += 1

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

input = np.array(input)

labelencoder = LabelEncoder()

y = input[:, 0]
input = input[:, 1:]
print(y)
print(input)
print(input[18])
#
# input = labelencoder.fit_transform(input)
# print(input)
input[:, 0] = labelencoder.fit_transform(input[:, 0])
input[:, 0] = [int(x) for x in input[:, 0]]
input[:, 1] = labelencoder.fit_transform(input[:, 1])
input[:, 1] = [int(x) for x in input[:, 1]]
input[:, 2] = labelencoder.fit_transform(input[:, 2])
input[:, 2] = [int(x) for x in input[:, 2]]
# #input[:, 3] = labelencoder.fit_transform(input[:, 3])
# input[:, 4] = labelencoder.fit_transform(input[:, 4])
input[:, 5] = labelencoder.fit_transform(input[:, 5])
input[:, 5] = [int(x) for x in input[:, 5]]
print(input)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import copy
input2 = copy.deepcopy(input)

classifier = RandomForestClassifier(n_estimators=5000, random_state=0)
classifier.fit(input, y)

# amazonia = 0
# concealed = 1
# echo = 2
# northren = 3
# refuge = 4
# swamped = 5
# terenas = 6
# turtle = 7
# twisted = 8
# --------------------------------------------- Input -----------------------------------------------------------
xin = [0, 1, 4, 60, 430, 2]
to_print = copy.deepcopy(xin)
print(parse_x(to_print))
# Hum = 0
# Ne = 1
# Orc = 2
# Ra = 3
# Ud = 4

# games less than 20 = 1
# games less than 60 = 2
# games less than 120 = 3
# many games = 4



onehot_encoded = []

letter = [0 for _ in range(5)]
letter[xin[0]] = 1

for i in letter:
    onehot_encoded.append(i)

letter = [0 for _ in range(5)]
letter[xin[2]] = 1
for i in letter:
    onehot_encoded.append(i)

letter = [0 for _ in range(9)]
letter[xin[5]] = 1
for i in letter:
    onehot_encoded.append(i)

onehot_encoded = np.array(onehot_encoded)
onehot_encoded.flatten()

onehot_encoded = np.append(onehot_encoded, xin[1])
onehot_encoded = np.append(onehot_encoded, xin[3])
onehot_encoded = np.append(onehot_encoded, xin[4])

onehot_encoded.flatten()


print("manual one hot "+str(onehot_encoded))

# print(parse_one_hot(xin))

y_pred1 = classifier.predict_proba([xin])
print(xin)
print("Chanses that Grubby wins "+str(y_pred1[0][1]*100)+"%")

#   Without map prediction   ##############################################

print("")
print("Without map prediciton ")
input = input[:, :-1]
classifier.fit(input, y)
xin = xin[:-1]
y_pred2 = classifier.predict_proba([xin])
print("Chanses that Grubby wins "+str(y_pred2[0][1]*100)+"%")

print("")
print("One hot ")
print("")

# One Hot ############################################################

# amazonia = 0
# concealed = 1
# echo = 2
# northren = 3
# refuge = 4
# swamped = 5
# terenas = 6
# turtle = 7
# twisted = 8





#xin = [1., 0., 0., 0., 0.,      0., 1., 0., 0., 0.,   0., 0., 0., 1., 0., 0., 0., 0., 0.,       1,      89, 430]

# Hum = 0
# Ne = 1
# Orc = 2
# Ra = 3
# Ud = 4

onehotencoder = OneHotEncoder(categorical_features=[0, 2, 5])
onehot_input = onehotencoder.fit_transform(input2).toarray()

classifier2 = RandomForestClassifier(n_estimators=5000, random_state=0)
classifier2.fit(onehot_input, y)
xin = onehot_encoded
y_pred3 = classifier2.predict_proba([xin])
print("Chanses that Grubby wins "+str(round(y_pred3[0][1]*100))+"%")


xin2 =[xin[0],  xin[1], xin[2], xin[3], xin[4], xin[5], xin[6], xin[7], xin[8], xin[9], xin[19], xin[20], xin[21]]

input2 = input2[:, :-1]
onehotencoder = OneHotEncoder(categorical_features=[0,  2])
onehot_input = onehotencoder.fit_transform(input2).toarray()

print("")
print("Without map prediciton ")

classifier3 = RandomForestClassifier(n_estimators=5000, random_state=0)
classifier3.fit(onehot_input, y)
y_pred4 = classifier3.predict_proba([xin2])

print(parse_one_hot(xin))
print("Chanses that Grubby wins "+str(y_pred4[0][1]*100)+"%")


print(" pred map "+str(round(y_pred1[0][1]*100))+"%"+" pred 2 "+str(round(y_pred2[0][1]*100))+"%"+" onehot map "+
      str(round(y_pred3[0][1]*100))+"%"+" onehot 3 "+str(round(y_pred4[0][1]*100))+"%")

print(parse_x(to_print))
s = parse_x(to_print)
print(s+"-"+str(int(round(y_pred1[0][1]*100)))
      +"%-"+str(int(round(y_pred2[0][1]*100)))
      +"%-"+str(int(round(y_pred3[0][1]*100)))
      +"%-"+str(int(round(y_pred4[0][1]*100)))+"%-")