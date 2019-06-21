from operator import add
import numpy as np
from matplotlib import pyplot as plt
import os


def excluded(i, excluded):
    for e in excluded:
        if i == e:
            return True


points = []
points.append(0)
def calc_scores(preds, participations, pred_number, excluded_list=[], cap=95):
    z = preds.split("-")
    #print(z)
    predictions = []

    for s in z:
        if '%' in s:
            s = s.replace('%', '')
            s = s.replace('\n', '')

            predictions.append(int(s))

    for i in range(len(predictions), n_predictions):
        predictions.append(0)

    result = int(z[0])
    #print("result "+str(result))

    s = n_predictions * [0]

    for i in range(0, n_predictions-1):
        if pred_number < FIXED_INPUT:
            if i > 13:
                predictions[i] = 0

        if excluded(i, excluded_list):
            continue
        if predictions[i] > cap:
            predictions[i] = cap

        if int(z[5]) <= 40:
            predictions[7] = 0


        if predictions[i] == 0:
            continue

        opponent = i+1
        for j in range(opponent, n_predictions):
            if excluded(j, excluded_list):
                continue

            if predictions[j] > cap:
                predictions[j] = cap

            if pred_number < FIXED_INPUT:
                if j > 13:
                    predictions[j] = 0

            if int(z[5]) <= 40:
                predictions[7] = 0

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
filepath = '../logs/predictions.txt'
model_to_train = os.path.join(script_directory, filepath)
f = open(filepath, "r")
contents = f.readlines()

n_predictions = 21
participations = n_predictions * [0]

counter = 0
cap = 95


print("cap "+str(cap))

participants = {0: "numerical rf", 1: "no maps numeric rf", 2: "one hot rf", 4: "logistic mu CV", 5: "logistic matchup",
                6: "normal logistic", 7: "strong logistic", 8: "transformed winrates rf", 9: "Vagelis", 10: "Egw",
                12: "winrates logistic", 13: "formula winrates logistic", 14: "neural1", 15: "neural2", 16: "neural3L3W",
                17: "neural4L-3W", 18: "neural4L4W", 19: "neural average", 20: "neural Cross"}

BALANCED = 468
STRONG_LONG_NO_BALANCED = 530
FIXED_INPUT = 563
STOP_RF_FROM_OVERFITTING = 641
LOGISTIC_MU_CV = 675


LIMIT = 642

opp = 9
graph_a =0
graph_b = 7


def calc_scores_vs_opponent(opponent, cap=95):
    scores_vs_opponent = n_predictions * [0]
    for participant in range(n_predictions):

        if participant == opponent:
            continue

        total_scores = n_predictions * [0]
        total_scores = np.array(total_scores)
        participations = n_predictions * [0]

        excluded = list(range(0, n_predictions+1))

        excluded.remove(participant)
        excluded.remove(opponent)

        counter = 0
        for i in contents:
            if counter > LIMIT:
                s = calc_scores(i, participations, counter, excluded, cap)
                s = np.array(s)

                if counter<675:
                    s[4] = 0

                total_scores = s + total_scores

            counter += 1

        if participations[participant] == 0:
            continue

        scores_vs_opponent[participant] = (total_scores[participant] / participations[participant])

        if participant == 1 or participant == 3  or participant == 8 or participant == 11 or participant == 12 or participant ==13:
            continue
        print(participants[participant] + " vs " + participants[opponent] + " " + str(scores_vs_opponent[participant]) +" se " + str(participations[participant]))

    return scores_vs_opponent


scores_vs_opp = calc_scores_vs_opponent(opp, 92)
# for p in range(n_predictions):
#     if p == 1 or p == 3 or p == 11:
#         continue
#
#     print(participants[p] + " vs " + participants[opp] + " " + str(scores_vs_opp[p]))

total_scores = n_predictions * [0]
total_scores = np.array(total_scores)
counter = 0
participations = n_predictions * [0]

exc = [1, 8, 4, 12,13, 3]
points = []
points.append(0)
for i in contents:
    if counter > LIMIT:
        s = calc_scores(i, participations, counter, exc)
        s = np.array(s)
        if counter < 675:
            s[4] = 0
        total_scores = s + total_scores

    counter += 1

print(points)
plt.plot(points)
plt.title(participants[graph_a]+" vs " + participants[graph_b])
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