from operator import add
import numpy as np

def calc_scores(preds):
    z = preds.split("-")
    print(z)
    predictions = []

    for s in z:
        if '%' in s:
            s = s.replace('%','')
            predictions.append(int(s))

    print(predictions)
    # predictions = [int(z[0]), int(z[1]), int(z[2]), int(z[3])]
    #
    # print(predictions)
    result = int(z[0])
    print("result "+str(result))

    if len(predictions) < 4:
        print("not enough predictions")
        return [0, 0, 0, 0]

    s = [0, 0, 0, 0, 0]

    for i in range(0, 4):
        z = i+1
        for j in range(z, 5):
            if predictions[i] == predictions[j]:
                continue

            # if i != 4:
            #     predictions[i] -= 2
            #
            # if j != 4:
            #     predictions[j] -= 2
            # else:
            #     predictions[j] += 2

            if result == 1:
                if predictions[i] > predictions[j]:
                    if predictions[i] > 50:
                        value = 1
                        s[i] += value
                        s[j] -= value
                    else:
                        value = (100 - predictions[i]) / predictions[i]
                        s[i] += value
                        s[j] -= value
                else:
                    if predictions[j] > 50:
                        value = 1
                        s[i] -= value
                        s[j] += value
                    else:
                        value = (100 - predictions[j]) / predictions[j]
                        s[i] -= value
                        s[j] += value
            else:
                if predictions[i] > predictions[j]:
                    if predictions[i] > 50:
                        value = predictions[i] / (100 - predictions[i])
                        s[i] -= value
                        s[j] += value
                    else:
                        value = 1
                        s[i] += value
                        s[j] -= value
                else:
                    if predictions[j] > 50:
                        value = predictions[j] / (100 - predictions[j])
                        s[i] += value
                        s[j] -= value
                    else:
                        value = 1
                        s[i] += value
                        s[j] -= value

    if sum(s) > 0.00001:
        print("error no zero sum")
        return [0, 0, 0, 0]
    else:
        print(s)

    return s


f = open("predictions.txt", "r")
contents = f.readlines()

total_scores = [0,0,0,0,0]
total_scores = np.array(total_scores)
for i in contents:
    s = calc_scores(i)
    s = np.array(s)
    total_scores = s + total_scores
    print("total")
    print(total_scores)
    print("")
