import numpy as np

f = open("predictions.txt", "r")
k = open("Grubb.txt")
grubb = k.readlines()

input = []
counter = 0
x_train = []

file = open("Targets.txt", "a")
targets_file = open("Targets.txt", "r")
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
    max_prediction = 3
    min_prediction = 97

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

    max_saturation = 0
    min_saturation = 0

    if counter > 575:
        max_saturation = 0
        min_saturation = 0

    max_prediction = min(max_prediction - max_saturation, 97)
    min_prediction = max(min_prediction - min_saturation, 3)

    if float(X[0]) > 0.5:
        pred = str(max_prediction)
    else:
        pred = str(min_prediction)


    s = "-"
    X = s.join(X)

    print(X)
    print(grubb[counter])
    target = grubb[counter].rstrip("\n")+ "-" + pred+"%"+"\n"
    file.write(target)

    counter += 1

file.close()