import numpy as np
from sklearn.preprocessing import LabelEncoder

f = open("Grubb.txt", "r")
contents = f.readlines()

input = []
counter = 0
x_train = []

file = open("GrubbC.csv", "a")

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
input[:, 1] = labelencoder.fit_transform(input[:, 1])
input[:, 1] = [int(x) for x in input[:, 1]]

input[:, 2] = labelencoder.fit_transform(input[:, 2])
input[:, 2] = [int(x) for x in input[:, 2]]

input[:, 3] = labelencoder.fit_transform(input[:, 3])
input[:, 3] = [int(x) for x in input[:, 3]]

input[:, 6] = labelencoder.fit_transform(input[:, 6])
input[:, 6] = [int(x) for x in input[:, 6]]

for element in input:
    file.write(str(element)+"\n")


file.close()