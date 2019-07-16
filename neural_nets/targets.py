import numpy as np

k = open("../logs/Grubb.txt")
grubb = k.readlines()
old_preds = []
for i in grubb:
    z = i.split("-")
    if int(z[4]) < 55:
        continue
    old_preds.append(i)

f = open("../logs/refinedPredictions.txt", "r")
contents = f.readlines()
automag = []
for i in contents:
    asd = i.split("-")
    if int(asd[4]) < 55:
        continue
    automag.append(i)

print(len(automag))
print(len(old_preds))
input = []
counter = 0
x_train = []

file = open("../logs/third_gen_targets.txt", "a")
targets_file = open("../logs/third_gen_targets.txt", "r")
targets = targets_file.readlines()

data = []

print(len(targets))

counter = 0
for line in automag:
    print("hi")
    X = line.split('-')

    grub_X = old_preds[counter].split('-')

    print(counter)

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

            if pred > 0:
                if pred < min_prediction:
                    min_prediction = pred

    max_saturation = 0
    min_saturation = 0

    max_prediction = min(max_prediction - max_saturation, 98)
    min_prediction = max(min_prediction - min_saturation, 3)

    if float(X[0]) > 0.5:
        pred = str(max_prediction)
    else:
        pred = str(min_prediction)


    s = "-"
    X = s.join(X)

    print(X)
    print(old_preds[counter])
    target = old_preds[counter].rstrip("\n") + "-" + pred+"%"+"\n"
    file.write(target)

    counter += 1

file.close()