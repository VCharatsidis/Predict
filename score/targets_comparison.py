from operator import add
import numpy as np
from matplotlib import pyplot as plt
import os



points = []
points.append(0)
def calc_scores(preds, participations,  cap=95):
    z = preds[0].split("-")
    z2 = preds[1].split("-")
    predictions = []

    for s in z:
        if '%' in s:
            s = s.replace('%', '')
            s = s.replace('\n', '')

            predictions.append(int(s))

    for s in z2:
        if '%' in s:
            s = s.replace('%', '')
            s = s.replace('\n', '')

            predictions.append(int(s))

    for i in range(len(predictions), n_predictions):
        predictions.append(0)

    result = int(z[0])
    #print("result "+str(result))

    s = n_predictions * [0]

    # if counter < 676:
    #     predictions[1] = 0
    #     predictions[3] = 0
    #     predictions[4] = 0

    for i in range(0, n_predictions-1):

        # if int(z[5]) <= 40:
        #     predictions[7] = 0
        #     predictions[3] = 0
        #     predictions[4] = 0

        # if pred_number < FIXED_INPUT:
        #     if i > 13:
        #         predictions[i] = 0

        if predictions[i] == 0:
            continue


        if predictions[i] > cap:
            predictions[i] = cap


        opponent = i+1
        for j in range(opponent, n_predictions):



            if predictions[j] > cap:
                predictions[j] = cap

            # if pred_number < FIXED_INPUT:
            #     if j > 13:
            #         predictions[j] = 0
            #
            # if int(z[5]) <= 40:
            #     predictions[7] = 0
            #     predictions[3] = 0
            #     predictions[4] = 0

            if predictions[j] == 0:
                continue

            if predictions[i] == predictions[j]:
                continue

            participations[i] += 1
            participations[j] += 1
            val = 0
            if result == 1:
                if predictions[i] > predictions[j]:
                    if predictions[i] > 50:
                        value = 1
                        val = value
                        s[i] += value
                        s[j] -= value
                    else:
                        value = (100 - predictions[i]) / predictions[i]
                        val = value
                        s[i] += value
                        s[j] -= value
                else:
                    if predictions[j] > 50:
                        value = 1
                        val = - value
                        s[i] -= value
                        s[j] += value
                    else:
                        value = (100 - predictions[j]) / predictions[j]
                        val = - value
                        s[i] -= value
                        s[j] += value
            else:
                if predictions[i] > predictions[j]:
                    if predictions[i] > 50:
                        value = predictions[i] / (100 - predictions[i])
                        val = - value
                        s[i] -= value
                        s[j] += value
                    else:
                        value = 1
                        val = - value
                        s[i] -= value
                        s[j] += value
                else:
                    if predictions[j] > 50:
                        value = predictions[j] / (100 - predictions[j])
                        val = value
                        s[i] += value
                        s[j] -= value
                    else:
                        value = 1
                        val = value
                        s[i] += value
                        s[j] -= value


            if i == graph_a and j==graph_b:
                points.append(points[-1] + val)

    if sum(s) > 0.00001:
        print("error no zero sum")
        return n_predictions * [0]

    return s


script_directory = os.path.split(os.path.abspath(__file__))[0]
filepath = '../logs/third_gen_targets.txt'
model_to_train = os.path.join(script_directory, filepath)
f = open(filepath, "r")
contents = f.readlines()

filepath2 = '../logs/refinedTargets.txt'
model_to_train2 = os.path.join(script_directory, filepath2)
f2 = open(filepath2, "r")
contents2 = f2.readlines()

old_preds = []
for i in contents2:
    z = i.split("-")
    if int(z[4]) < 55:
        continue
    old_preds.append(i)

n_predictions = 21
participations = n_predictions * [0]

counter = 0
cap = 95


print("cap "+str(cap))

participants = {0: "numerical rf", 1: "old numeric", 2: "one hot rf", 3: "strong logistic CV", 4: "logistic mu CV", 5: "logistic matchup",
                6: "normal logistic", 7: "strong logistic", 8: "transformed winrates rf", 9: "Vagelis", 10: "Egw",
                12: "winrates logistic", 13: "formula winrates logistic", 14: "neural1", 15: "neural2", 16: "neural3L3W",
                17: "neural4L-3W", 18: "neural4L4W", 19: "neural average", 20: "neural Cross"}

BALANCED = 468
STRONG_LONG_NO_BALANCED = 530
FIXED_INPUT = 563
STOP_RF_FROM_OVERFITTING = 641
LOGISTIC_MU_CV = 676


LIMIT = -1
UPPER_LIMIT = 2000

opp = 9
graph_a = 9
graph_b = 20



total_scores = n_predictions * [0]
total_scores = np.array(total_scores)

participations = 2 * [0]

exc = [8, 12, 13]
points = []
points.append(0)


for i in range(len(contents)):

    if i > LIMIT and i < UPPER_LIMIT:
        print(contents2[i])
        print(contents[i])
        preds = [contents[i], contents2[i]]
        s = calc_scores(preds, participations)
        s = np.array(s)

        total_scores = s + total_scores
        points.append(total_scores[0])

    counter += 1

print(points)
plt.plot(points)
plt.title("third_gen_targets"+" vs " + "refinedTargetsSecond")
plt.ylabel('winnings '+ participants[graph_a])
plt.xlabel('bets')
plt.show()
print("participations")
print(participations)
print("total means")
participations = np.array(participations)
total_scores = total_scores/(participations+1)
print(total_scores)
print("cap "+str(cap))

# result_dict = {"strong: ": int(round(strong_logistic * 100)), "p_log_reg formula: ": int(round(preprocessed_logreg_formula*100)),
#                "p_log_reg: ": int(round(preprocessed_logreg_no_formula * 100)), "rf_winrates formula: ": rf_trasformed,
#                "rf_winrates: ": predComboLeanring, "matchups logistic: ": logistic_mutchups, "normal logistic: ":logistic_pred,
#                "one hot rf: ": pred3}