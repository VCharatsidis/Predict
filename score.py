from operator import add
import numpy as np

n_predictions = 14
participations = n_predictions * [0]
cap = 95

def excluded(i, excluded):
    for e in excluded:
        if i == e:
            return True

last_predictions = -1
def calc_scores(preds):
    z = preds.split("-")
    print(z)
    predictions = []

    for s in z:
        if '%' in s:
            s = s.replace('%', '')
            s = s.replace('\n', '')
            predictions.append(int(s))

    for i in range(len(predictions), n_predictions):
        predictions.append(0)

    result = int(z[0])
    print("result "+str(result))

    s = n_predictions * [0]
    excluded_list = []
    excluded_list = [2, 0, 1, 4, 3, 5, 6, 7, 8]

    for i in range(0, n_predictions-1):

        if excluded(i, excluded_list):
            continue
        if predictions[i] > cap:
            predictions[i] = cap

        if predictions[i] == 0:
            continue

        if int(z[5]) <= 70:
            predictions[7] = 0

        opponent = i+1
        for j in range(opponent, n_predictions):
            if excluded(j, excluded_list):
                continue
            # if j < 5:
            #     continue
            # if j > 7:
            #     continue
            if predictions[j] > cap:
                predictions[j] = cap

            if predictions[j] == 0:
                continue

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
counter = 0
for i in contents:
    if counter > last_predictions:
        s = calc_scores(i)
        s = np.array(s)
        total_scores = s + total_scores
        print("total")
        print(total_scores)
        print("")
    counter += 1

print("participations")
print(participations)
print("total means")
participations = np.array(participations)
total_scores = total_scores/(participations+1)
print(total_scores)
print("cap "+str(cap))