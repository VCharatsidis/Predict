from operator import add
import numpy as np
from matplotlib import pyplot as plt
import os
import statistics
from validations_ids import get_validation_ids

def excluded(i, excluded):
    for e in excluded:
        if i == e:
            return True


points = []
points.append(0)
def calc_scores(vagelis, egw, counter, preds, participations, pred_number, excluded_list=[], cap=97):
    z = preds.split("-")
    predictions = []

    for s in z:
        if '%' in s:
            s = s.replace('%', '')
            s = s.replace('\n', '')

            predictions.append(int(s))


    predictions[9] = vagelis
    predictions[10] = egw

    # if predictions[23] > 66:
    #     predictions[0] = predictions[23]
    # else:
    #     predictions[0] = predictions[15]
    #
    # predictions[1] = predictions[0] * 0.5 + predictions[23] * 0.5
    # predictions[2] = predictions[0] * 0.4 + predictions[23] * 0.6
    #
    # predictions[3] = predictions[0] * 0.6 + predictions[23] * 0.4
    #
    # predictions[4] = predictions[14] * 0.6 + predictions[21] * 0.4
    # predictions[5] = predictions[0] * 0.7 + predictions[23] * 0.3
    # predictions[6] = predictions[14] * 0.5 + predictions[23] * 0.5
    # predictions[7] = predictions[15] * 0.7 + predictions[21] * 0.3
    # predictions[8] = predictions[0] * 0.5 + predictions[23] * 0.5

    # for i in range(len(predictions), n_predictions):
    #     predictions.append(0)

    result = int(z[0])

    s = n_predictions * [0]

    for i in range(0, len(predictions)-1):

        if predictions[i] == 0:
            continue

        if excluded(i, excluded_list):
            continue

        if predictions[i] > cap:
            predictions[i] = cap

        opponent = i+1
        for j in range(opponent, len(predictions)-1):

            if excluded(j, excluded_list):
                continue

            if predictions[j] > cap:
                predictions[j] = cap

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
filepath = '../logs/automagic.txt'
model_to_train = os.path.join(script_directory, filepath)
f = open(filepath, "r")
contents = f.readlines()

filepath2 = '../logs/automagic.txt'
model_to_train2 = os.path.join(script_directory, filepath2)
f2 = open(filepath2, "r")
contents2 = f2.readlines()

old_preds = []
for i in contents2:
    z = i.split("-")
    old_preds.append(i)

n_predictions = 27
participations = n_predictions * [0]

counter = 0
cap = 95


print("cap "+str(cap))

participants = {0: "numerical rf", 1: "old numeric", 2: "one hot rf", 3: "strong logistic CV", 4: "logistic mu CV",
                5: "logistic matchup",
                6: "normal logistic", 7: "strong logistic", 8: "transformed winrates rf", 9: "Vagelis", 10: "Egw",
                12: "merged cros4 pred1", 13: "average", 14: "neural1", 15: "neural2", 16: "neural3L3W",
                17: "neural4L-3W", 18: "neural4L4W", 19: "neural average", 20: "neural Cross", 21: "neural C 2",
                22: "neural C 3", 23: "neural C 4", 25: "metamodel", 26:"sigmamodel"}

BALANCED = 468
STRONG_LONG_NO_BALANCED = 530
FIXED_INPUT = 563
STOP_RF_FROM_OVERFITTING = 641
LOGISTIC_MU_CV = 676
NO_HERO = 924
LAST_TRAINED = 1009


LIMIT = 800
UPPER_LIMIT = 3000

opp = 10
graph_a = 10
graph_b = 18

val_ids = get_validation_ids()
val_ids = val_ids[-200:]
print(val_ids)
val_ids = list(range(LIMIT, UPPER_LIMIT))


def calc_scores_vs_opponent(opponent, cap=95):
    scores_vs_opponent = n_predictions * [0]
    for participant in range(n_predictions):

        if participant == opponent:
            continue

        total_scores = n_predictions * [0]
        total_scores = np.array(total_scores)
        participations = n_predictions * [0]

        excluded = list(range(0, n_predictions))

        excluded.remove(participant)
        excluded.remove(opponent)

        counter = 0
        for i in contents:
            if counter in val_ids:
                humans = old_preds[counter].split("-")

                my_spot = 17
                vag_spot = 16
                if counter > NO_HERO:
                    my_spot += 1
                    vag_spot += 1

                if len(humans) < 18:
                    vagelis = 0
                    egw = 0
                else:
                    humans[vag_spot] = humans[vag_spot].replace('%', '')
                    humans[my_spot] = humans[my_spot].replace('%', '')
                    if '\n' in humans[my_spot]:
                        humans[my_spot] = humans[my_spot].replace('\n', '')

                    vagelis = int(humans[vag_spot])

                    if humans[my_spot] =='':
                        egw = 0
                    else:
                        egw = int(humans[my_spot])

                # vagelis = 0
                # egw = 0
                s = calc_scores(vagelis, egw, counter, i, participations, counter, excluded)
                s = np.array(s)

                total_scores = s + total_scores

            counter += 1

        if participations[participant] == 0:
            continue

        scores_vs_opponent[participant] = (total_scores[participant] / participations[participant])

        print(participants[participant] + " vs " + participants[opponent] + " " + str(scores_vs_opponent[participant]) + " se " + str(participations[participant]))

    return scores_vs_opponent


scores_vs_opp = calc_scores_vs_opponent(opp, 97)


total_scores = n_predictions * [0]
total_scores = np.array(total_scores)
counter = 0
participations = n_predictions * [0]

exc = []
points = []
points.append(0)
NO_HERO = 924

for i in contents:
    if counter in val_ids:
        humans = old_preds[counter].split("-")

        my_spot = 17
        vag_spot = 16
        if counter > NO_HERO:
            my_spot += 1
            vag_spot += 1

        if len(humans) < 18:
            vagelis = 0
            egw = 0
        else:

            humans[vag_spot] = humans[vag_spot].replace('%', '')
            humans[my_spot] = humans[my_spot].replace('%', '')
            if '\n' in humans[my_spot]:
                humans[my_spot] = humans[my_spot].replace('\n', '')

            vagelis = int(humans[vag_spot])
            if humans[my_spot] == '':
                egw = 0
            else:
                egw = int(humans[my_spot])

        # if counter % 100 == 0:
        #     print("vagelis " + str(vagelis))
        #     print("egw " + str(egw))
        #     print(counter)
        #     print(humans)
        #     print(i)

        # vagelis = 0
        # egw = 0
        s = calc_scores(vagelis, egw, counter, i, participations, counter, exc)
        s = np.array(s)

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