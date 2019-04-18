from operator import add
import numpy as np

n_predictions = 11
participations = n_predictions * [0]

def calc_scores(preds):
    z = preds.split("-")
    print(z)
    predictions = []

    for s in z:
        if '%' in s:
            s = s.replace('%', '')
            s = s.replace('\n', '')
            predictions.append(int(s))

    for i in range (len(predictions), n_predictions):
        predictions.append(0)

    result = int(z[0])
    print("result "+str(result))

    s = n_predictions * [0]

    if n_predictions < 4:
        print("not enough predictions")
        return s


    for i in range(0, n_predictions-1):
        if predictions[i] == 0:
            continue

        # if predictions[i] > 80 or predictions[i] < 25:
        #     continue

        z = i+1
        for j in range(z, n_predictions):
            if predictions[j] == 0:
                continue

            # if predictions[j] > 80 or predictions[j] < 25:
            #     continue

            participations[i] += 1
            participations[j] += 1

            if predictions[i] == predictions[j]:
                continue

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
                        s[i] -= value
                        s[j] += value
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
        return n_predictions * [0]
    else:
        print(s)

    return s


f = open("predictions.txt", "r")
contents = f.readlines()

total_scores = n_predictions * [0]
total_scores = np.array(total_scores)
for i in contents:
    s = calc_scores(i)
    s = np.array(s)
    total_scores = s + total_scores
    print("total")
    print(total_scores)
    print("")

print("participations")
print(participations)
print("total means")
participations = np.array(participations)
total_scores = total_scores/participations
print(total_scores)
