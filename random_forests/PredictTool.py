import numpy as np
from sklearn.ensemble import RandomForestClassifier
import copy
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from logistic_reggresions.logistic_mutchups import logistic_reg
from logistic_reggresions import strong_logistic
from load_models import load_models

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)
#simplefilter(action='ignore', category="SourceChangeWarning")

import config


def parse_x(xin, result):

    data = str(result)+"-"

    data += config.races[xin[0]]+"-"
    data += config.tryhard[xin[1]]
    data += config.races[xin[2]]+"-"
    data += str(xin[3])
    data += "-"+str(xin[4])+"-"

    if len(xin)>5:
        data += config.maps[xin[5]]

    return data


avg_opponents_winrate = 0
observed_grubby_wins = {'Hum': 0, 'Ne': 0, 'Orc': 0, 'Ra': 0, 'Ud': 0}
observer_grubby_games = {'Hum': 0, 'Ne': 0, 'Orc': 0, 'Ra': 0, 'Ud': 0}
observed_grubby_winrates = {'Hum': 0, 'Ne': 0, 'Orc': 0, 'Ra': 0, 'Ud': 0}

wins_less_than_60 = 0
games_less_than_60 = 0


def predict(input_cp, original_input_for_strong_log_reg, y, i):
    xin = copy.deepcopy(input_cp[i])
    xin = [int(x) for x in xin]

    input_cp = np.delete(input_cp, i, axis=0)
    #print(len(input_cp))

    result = y[i]
    y = np.delete(y, i, axis=0)




    original_input = copy.deepcopy(input_cp)

    labelencoder = LabelEncoder()


    input2 = copy.deepcopy(input_cp)
    # # #
    # # # # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Estimators ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # errors = []
    # #
    # old_numeric_rf = RandomForestClassifier(n_estimators=1000, min_samples_split=10, oob_score=True)
    # old_numeric_rf.fit(input_cp, y)
    #
    # numeric_rf = RandomForestClassifier(n_estimators=1000,
    #                                     random_state=0,
    #                                     oob_score=True,
    #                                     min_samples_leaf=18,
    #                                     min_samples_split=5,
    #                                     max_features=3,
    #                                     max_depth=20,
    #                                     bootstrap=True)
    # numeric_rf.fit(input_cp, y)
    # oob_error1 = 1 - numeric_rf.oob_score_
    # errors.append(oob_error1)
    # importances1 = numeric_rf.feature_importances_
    #
    #
    #
    # np.set_printoptions(precision=2)
    # importances1 = ['%.2f'%(float(a)) for a in importances1]
    #
    #
    # #logistic_mutchups, logistic_mu_CV = logistic_reg(xin, "../logs/Grubb.txt", i)
    #
    #
    # # Hum = 0
    # # Ne = 1
    # # Orc = 2
    # # Ra = 3
    # # Ud = 4
    #
    # to_print = copy.deepcopy(xin)

    #print(parse_x(to_print, result))

    to_print = copy.deepcopy(xin)
    onehot_encoded = []

    letter = [0 for _ in range(5)]
    letter[xin[0]] = 1
    for i in letter:
        onehot_encoded.append(i)

    letter = [0 for _ in range(2)]
    letter[xin[1]] = 1
    for i in letter:
        onehot_encoded.append(i)

    letter = [0 for _ in range(5)]
    letter[xin[2]] = 1
    for i in letter:
        onehot_encoded.append(i)

    letter = [0 for _ in range(config.map_number)]
    letter[xin[5]] = 1
    for i in letter:
        onehot_encoded.append(i)

    onehot_encoded = np.array(onehot_encoded)
    onehot_encoded.flatten()

    onehot_encoded = np.append(onehot_encoded, xin[3])
    onehot_encoded = np.append(onehot_encoded, xin[4])

    onehot_encoded.flatten()

    neural_pred, neural_pred2, neural_pred3L3W, neural_pred4L3W, neural_pred4L4W,\
    neural_predCross, neural_predCross2, neural_predCross3,\
    neural_predCross4, neural_meta, coeffs = load_models(onehot_encoded)




    # 22222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222

    # numeric_pred = numeric_rf.predict_proba([xin])
    # old_numeric_pred = old_numeric_rf.predict_proba([xin])
    # # #
    # # #
    # # # # One Hot ############################################################
    # # #
    # # #
    # onehotencoder = OneHotEncoder(categorical_features=[0, 1, 2, 5])
    # onehot_input = onehotencoder.fit_transform(input2).toarray()
    # logistic_input = copy.deepcopy(onehot_input)
    # # # #
    # # # #
    #
    # # # #
    # # # #
    # # # # ############################################################## one hot rf ###############################
    # # # #
    # one_hot_rf = RandomForestClassifier(n_estimators=1000,
    #                                     random_state=0,
    #                                     oob_score=True,
    #                                     min_samples_leaf=15,
    #                                     min_samples_split=29,
    #                                     max_features=21,
    #                                     max_depth=20,
    #                                     bootstrap=True)
    #
    # one_hot_rf.fit(onehot_input, y)
    # oob_error3 = 1 - one_hot_rf.oob_score_
    # errors.append(oob_error3)
    #
    # xin = onehot_encoded
    # oh_rf_pred = one_hot_rf.predict_proba([xin])
    #
    # importances2 = one_hot_rf.feature_importances_
    #
    #

    #
    #
    s = parse_x(to_print, result)
    #

    # logistic_mutchups = int(round(logistic_mutchups[0][1]*100))
    # logistic_mu_CV = int(round(logistic_mu_CV[0][1]*100))


    old_numeric_pred = 0
    pred1 = 0
    pred3 = 0
    logistic_pred = 0
    logistic_mutchups = 0
    logistic_mu_CV = 0
    strong_logistic = 0
    strong_logistic_CV = 0

    # neural_pred[0][0] = 0
    # neural_pred2[0][0] = 0
    # neural_pred3L3W[0][0] = 0
    # neural_pred4L3W[0][0] = 0
    # neural_pred4L4W[0][0] = 0
    # neural_predTest[0][0] = 0
    #
    # neural_predCross2[0][0] = 0
    # neural_predCross3[0][0] = 0

    s = parse_x(to_print, result)
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ RESULT #############################################################################

    if neural_predCross4[0][0] * 100 > 66:
        merged = neural_predCross2[0][0] * 98
    else:
        merged = neural_pred3L3W[0][0] * 102

    merged = (neural_pred3L3W[0][0]*0.5 + neural_predCross2[0][0]*0.5) * 100

    averaged = (neural_pred[0][0] + neural_pred2[0][0] + neural_pred3L3W[0][0] + neural_pred4L3W[0][0] +
                neural_pred4L4W[0][0]
                + neural_predCross[0][0] + neural_predCross2[0][0] + neural_predCross3[0][0] + neural_predCross4[0][
                    0]) / 9 * 100
    #
    if neural_predCross4[0][0] > 0.66:
        a = neural_predCross4[0][0] * 98
    else:
        a = neural_pred2[0][0] * 102
    #
    # old_numeric_pred = pred1 * 0.5 + neural_predCross4[0][0] * 98 * 0.5
    # pred3 = pred1 * 0.4 + neural_predCross[0][0] * 100 * 0.6
    #
    #
    # if neural_predCross[0][0] > 0.66:
    #     another_merge = neural_predCross[0][0] * 98
    # else:
    #     another_merge = neural_pred[0][0] * 102
    #
    # strong_logistic_CV = another_merge * 0.6 + neural_predCross2[0][0] * 102 * 0.4
    #
    # logistic_mu_CV = neural_pred[0][0] * 100 * 0.6 + neural_predCross4[0][0] * 99 * 0.4
    # logistic_mutchups = pred1 * 0.7 + neural_predCross3[0][0] * 100 * 0.3
    # logistic_pred = neural_pred[0][0] * 100 * 0.5 + neural_predCross4[0][0] * 100 * 0.5
    # strong_logistic = neural_pred2[0][0] * 100 * 0.5 + neural_predCross2[0][0] * 100 * 0.5
    another = a * 0.7 + neural_predCross[0][0] * 100 * 0.3


    log =(s + "-" + str(int(round(pred1)))
          + "%-" + str(int(round(0)))
          + "%-" + str(int(round(0)))
          + "%-" + str(int(round(strong_logistic_CV)))
          + "%-" + str(int(round(logistic_mu_CV)))
          + "%-" + str(int(round(logistic_mutchups)))
          + "%-" + str(int(round(logistic_pred)))
          + "%-" + str(int(round(strong_logistic)))
          + "%-" + str(int(round(another)))+"%-"
          + "0%-"
          + "0%-"
          + "0%-"
          + str(int(round(merged))) +"%-"
          + str(int(round(averaged)))+"%"
          + "-" + str((int(round(neural_pred[0][0]*100)))) + "%"
          + "-" + str((int(round(neural_pred2[0][0]*100)))) + "%"
          + "-" + str((int(round(neural_pred3L3W[0][0] * 100))))+"%"
          + "-" + str((int(round(neural_pred4L3W[0][0] * 100))))+"%"
          + "-" + str((int(round(neural_pred4L4W[0][0] * 100))))+"%"
          + "-0%"
          + "-" + str((int(round(neural_predCross[0][0]*100)))) + "%"
          + "-" + str((int(round(neural_predCross2[0][0] * 100)))) + "%"
          + "-" + str((int(round(neural_predCross3[0][0] * 100)))) + "%"
          + "-" + str((int(round(neural_predCross4[0][0] * 100)))) + "%"
          + "-0%-"
          + str(int(round(neural_meta[0]*100))) + "%"
          + "\n")

    print(log)


################################## Score between classifiers ################################

    file = open("../logs/new_predictions.txt", "a")
    file.write(log)
    file.close()

    return coeffs




def prepare_input():
    f = open("../logs/Grubb.txt", "r")
    contents = f.readlines()

    for line in contents:
        input_cp = []
        counter = 0

        for line in contents:
            X = line.split('-')

            X[4] = int(X[4])

            X[5] = int(X[5])

            if "\n" in X[6]:
                X[6] = X[6].rstrip("\n")

            X = np.array(X)

            input_x = []
            input_x.append(X[0])
            input_x.append(X[1])
            input_x.append(X[2])
            input_x.append(X[3])
            input_x.append(X[4])
            input_x.append(X[5])
            input_x.append(X[6])
            input_x = np.array(input_x)

            input_cp.append(input_x)

            counter += 1

        labelencoder = LabelEncoder()

        input_cp = np.array(input_cp)
        original_input_for_strong_log_reg = copy.deepcopy(input_cp)
        y = input_cp[:, 0]
        input_cp = input_cp[:, 1:]

        input_cp[:, 0] = labelencoder.fit_transform(input_cp[:, 0])
        input_cp[:, 0] = [int(x) for x in input_cp[:, 0]]

        input_cp[:, 1] = labelencoder.fit_transform(input_cp[:, 1])
        input_cp[:, 1] = [int(x) for x in input_cp[:, 1]]

        input_cp[:, 2] = labelencoder.fit_transform(input_cp[:, 2])
        input_cp[:, 2] = [int(x) for x in input_cp[:, 2]]

        input_cp[:, 5] = labelencoder.fit_transform(input_cp[:, 5])
        input_cp[:, 5] = [int(x) for x in input_cp[:, 5]]

        return input_cp, original_input_for_strong_log_reg, y


input, or_input, y = prepare_input()
print("data number "+str(len(input)))

coeffs_total = 9 * [float(0)]
coeffs_total = np.array(coeffs_total)
for i in range(len(input)):

    xin = input[i]

    coeffs = predict(input, or_input, y, i)
    coeffs_total += coeffs
    print(coeffs)

print("coeffs total")
print(coeffs_total/len(input))



