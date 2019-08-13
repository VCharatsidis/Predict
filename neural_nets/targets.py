import numpy as np

k = open("../logs/Grubb.txt")
grubb = k.readlines()
old_preds = []
for i in grubb:
    z = i.split("-")
    old_preds.append(i)

f = open("../logs/new_predictions.txt", "r")
contents = f.readlines()
automag = []
for i in contents:
    asd = i.split("-")
    automag.append(i)

print(len(automag))
print(len(old_preds))
input = []
counter = 0
x_train = []

file = open("../logs/gaussianPredictions.txt", "a")
targets_file = open("../logs/gaussianPredictions.txt", "r")
targets = targets_file.readlines()

data = []

print(len(targets))

counter = 0
other_counter = 0
negative_bonus = 0
positive_bonus = 0
for line in automag:
    print("hi")
    X = line.split('-')

    grub_X = old_preds[counter].split('-')

    print(counter)

    X = line.split('-')

    processed_X = []
    max_prediction = 3
    min_prediction = 98

    second_max = 3
    second_min = 98

    array_x = []

    for x in X:
        if '%' in x:
            x = x.rstrip("\n")
            x = x.replace('%', '')

            pred = int(x)
            if pred > 0:
                array_x.append(pred)


            # if pred > max_prediction:
            #     max_prediction = pred
            #
            # if pred > 0:
            #     if pred < min_prediction:
            #         min_prediction = pred


    array_x = np.array(array_x)
    array_x = array_x[8:12]

    mean = np.mean(array_x, axis=0)
    std = np.std(array_x, axis=0)

    neg_bonus = min(1.9 * std, 12)
    pos_bonus = min(1 * std, 12)

    max_prediction = array_x[0]
    min_prediction = array_x[0]

    max_prediction = int(round(mean + pos_bonus))
    min_prediction = int(round(mean - neg_bonus))

    # max_saturation = 0
    # min_saturation = 0
    #
    # max_prediction = min(max_prediction - max_saturation, 98)
    # min_prediction = max(min_prediction - min_saturation, 3)

    if float(X[0]) > 0.5:
        pred = str(max_prediction)
        positive_bonus += pos_bonus
        print("positive bonus " + str(positive_bonus))
    else:
        pred = str(min_prediction)
        negative_bonus -= neg_bonus
        print("negative bonus " + str(negative_bonus))

    s = "-"
    X = s.join(X)

    print(X)
    print(old_preds[counter])
    target = old_preds[counter].rstrip("\n") + "-" + pred+"%"+"\n"
    file.write(target)

    counter += 1

file.close()