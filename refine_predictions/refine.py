import numpy as np

f = open("../logs/predictions.txt", "r")
k = open("../logs/Grubb.txt")
grubb = k.readlines()

input = []
counter = 0
x_train = []

file = open("../logs/Targets.txt", "a")
targets_file = open("../logs/Targets.txt", "r")
targets = targets_file.readlines()

data = []

contents = f.readlines()
counter = 0
print(len(targets))
for line in contents:
    if counter < len(targets):
        counter += 1
        continue

    print("hi")
    X = line.split('-')

    processed_X = []


    for x in X:
        if '%' in x:
            x = x.rstrip("\n")
            x = x.replace('%', '')
            pred = int(x)

            if pred > max_prediction:
                max_prediction = pred

            if pred > 0:
                if pred < min_prediction:
                    min_prediction = pred


    max_prediction = min(max_prediction, 98)
    min_prediction = max(min_prediction, 3)

    if float(X[0]) > 0.5:
        pred = str(max_prediction)
    else:
        pred = str(min_prediction)

    s = "-"
    X = s.join(X)

    print(X)
    print(grubb[counter])
    target = grubb[counter].rstrip("\n") + "-" + pred+"%"+"\n"
    file.write(target)

    counter += 1

file.close()