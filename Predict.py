f = open("C:\\Users\chara\Desktop\Grubb.txt", "r")
import numpy as np
import pandas as pd
saturation = 0

def parse_x(xin):
    data = ""
    if xin[1] == saturation+1:
        data += "tryhard "
    else:
        data +="request "

    if xin[0] == saturation+1:
        data +="Ne"
    elif xin[0] == saturation+2:
        data +="Orc"
    elif xin[0] == saturation+3:
        data +="Random"
    elif xin[0] == saturation+4:
        data +="Undead"
    else:
        data +="Human"

    data +=" vs "

    if xin[2] == saturation+1:
        data +="Ne"
    elif xin[2] == saturation+2:
        data +="Orc"
    elif xin[2] == saturation+3:
        data +="Random"
    elif xin[2] == saturation+4:
        data +="Undead"
    else:
        data +="Human"

    data += " "
    data += str(xin[3])
    data += " games "

    if xin[4]== saturation+1:
        data += " less than 20"
    elif xin[4]== saturation+2:
        data += " less than 60"
    elif xin[4]== saturation+3:
        data += " less than 120"
    else:
        data += " many"

    # data +=" map "
    # data += str(xin[5])

    return data

def less_buckets(games):
    bucket = 0

    if games <= 20:
        bucket = saturation+1
    elif games <= 60:
        bucket = saturation+2
    elif games <= 150:
        bucket = saturation+3
    else:
        bucket = saturation+4

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
    z = int(X[4])
    stats_adipalou = (round(z/5))*5
    X[4] = stats_adipalou
    games_adipalou = int(X[5])
    b = less_buckets(games_adipalou)
    X[5] = b
    X[6] = X[6].rstrip("\n")


    X = np.array(X)
    input.append(X)

    print("bucket "+str(b))
    print("stats adipalou rounded "+str(stats_adipalou))
    print("stats adipalou "+str(z))
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
input[:, 0] = [int(x)+ saturation for x in input[:, 0]]
input[:, 1] = labelencoder.fit_transform(input[:, 1])
input[:, 1] = [int(x)+saturation for x in input[:, 1]]
input[:, 2] = labelencoder.fit_transform(input[:, 2])
input[:, 2] = [int(x)+saturation for x in input[:, 2]]
# #input[:, 3] = labelencoder.fit_transform(input[:, 3])
# input[:, 4] = labelencoder.fit_transform(input[:, 4])
input[:, 5] = labelencoder.fit_transform(input[:, 5])
input[:, 5] = [int(x)+saturation for x in input[:, 5]]
print(input)



# onehotencoder = OneHotEncoder(categorical_features=[0, 2, 5])
# input = onehotencoder.fit_transform(input).toarray()


from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
classifier = RandomForestClassifier(n_estimators=2000, random_state=0)
classifier.fit(input, y)

number = 54
print("number "+str(number) + " : "+str(input[number]))

print("result "+str(y[number]))

xin = [saturation+1, saturation+0, saturation+0, 70, saturation+4, saturation+6]
print("input "+str(xin))
print(parse_x(xin))
print(input[-1])
print("")

# xin = [[0., 0., 1., 0., 0.,      1., 0., 0., 0., 0.,   0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,  saturation+1,      85, saturation+4]]
# print("one hot in "+str(xin))

y_pred = classifier.predict_proba([xin])
print("Chanses that Grubby wins "+str(y_pred[0][1]*100)+"%")
print("probabilities "+str(y_pred[0]))


regressor = RandomForestRegressor(n_estimators=2000, random_state=0)
regressor.fit(input, y)
y_reg = regressor.predict([xin])
print("regressor "+str(y_reg))

#   Without map prediction   ##############################################

print("")
print("Without map prediciton ")
print("")
input = input[:, :-1]

xin = [0., 1., 0., 0., 0.,        1., 0.,        1., 0., 0., 0., 0.,          70, saturation+4]
print("one hot in "+str(xin))
onehotencoder = OneHotEncoder(categorical_features=[0, 1, 2])
input = onehotencoder.fit_transform(input).toarray()

print("number "+str(number)+" : "+str(input[number]))

classifier.fit(input, y)
regressor.fit(input, y)

# xin = xin[:-1]
# print("input without map"+str(xin))
# print(parse_x(xin))


y_pred = classifier.predict_proba([xin])
print("Chanses that Grubby wins "+str(y_pred[0][1]*100)+"%")
print("probabilities "+str(y_pred[0]))

y_reg = regressor.predict([xin])
print("regressor "+str(y_reg))