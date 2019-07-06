import numpy as np

f = open("../logs/refinedPredictions.txt", "r")
k = open("../logs/Grubb.txt")
grubb = k.readlines()

input = []
counter = 0
x_train = []

file = open("../logs/refinedTargetsSecond.txt", "a")
targets_file = open("../logs/refinedTargetsSecond.txt", "r")
targets = targets_file.readlines()

data = []

contents = f.readlines()
counter = 0
grub_counter = 0
print(len(targets))
for line in contents:

    grub_X = grubb[counter].split('-')
    if int(grub_X[4]) < 55:
        grub_counter += 1

    if counter < len(targets):
        counter += 1
        continue

    print("hi")
    X = line.split('-')

    processed_X = []
    max_prediction = 3
    min_prediction = 98

    second_max = 3
    second_min = 98

    for x in X:
        if '%' in x:
            x = x.rstrip("\n")
            x = x.replace('%', '')
            pred = int(x)

            if pred > max_prediction:
                max_prediction = pred

            if pred < max_prediction and pred > second_max:
                second_max = pred

            if pred > 0:
                if pred < min_prediction:
                    min_prediction = pred

                if pred > min_prediction and pred < second_min:
                    second_min = pred

    max_saturation = 0
    min_saturation = 0

    max_prediction = min(max_prediction - max_saturation, 98)
    min_prediction = max(min_prediction - min_saturation, 3)

    if float(X[0]) > 0.5:
        pred = str(second_max)
    else:
        pred = str(second_min)


    s = "-"
    X = s.join(X)

    print(X)
    print(grubb[counter + grub_counter])
    target = grubb[counter+grub_counter].rstrip("\n")+ "-" + pred+"%"+"\n"
    file.write(target)

    counter += 1

file.close()