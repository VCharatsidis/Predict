import numpy as np
from sklearn.ensemble import RandomForestClassifier
import copy
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from logistic_reggresions.logistic_mutchups import logistic_reg

from load_models import load_models
from xgboost import XGBClassifier

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

    # input2 = copy.deepcopy(input_cp)
    #
    # ######################### XGB ##############################################
    # # xgb = XGBClassifier(n_estimators=300, max_depth=20)
    # # xgb.fit(input_cp, y)
    # # xgb_pred = xgb.predict_proba([xin])
    # #
    # # # # #
    # # # # # # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Estimators ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # errors = []
    # # # #
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
    # logistic_mutchups, logistic_mu_CV = logistic_reg(xin, "../logs/Grubb.txt", i)
    # #
    # #
    # # # Hum = 0
    # # # Ne = 1
    # # # Orc = 2
    # # # Ra = 3
    # # # Ud = 4
    # #
    to_print = copy.deepcopy(xin)
    #
    print(parse_x(to_print, result))
    #
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
    neural_predCross4, neural_meta, coeffs, sigma, enhanced = load_models(onehot_encoded)
    #
    #
    #
    #
    # # 22222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222
    #
    #
    # # #
    # # #
    # # # # One Hot ############################################################
    # # #
    # # #
    # onehotencoder = OneHotEncoder(categorical_features=[0, 1, 2, 5])
    # onehot_input = onehotencoder.fit_transform(input2).toarray()
    # logistic_input = copy.deepcopy(onehot_input)
    #
    #
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
    # old_numeric_pred = old_numeric_rf.predict_proba([xin])
    # numeric_pred = numeric_rf.predict_proba([xin])
    #
    # xin = onehot_encoded
    # oh_rf_pred = one_hot_rf.predict_proba([xin])
    #
    #
    #
    # s = parse_x(to_print, result)
    #
    # numeric_pred = int(round(numeric_pred[0][1]*100))
    # old_numeric_pred = int(round(old_numeric_pred[0][1]*100))
    # oh_rf_pred = int(round(oh_rf_pred[0][1]*100))
    # logistic_mutchups = int(round(logistic_mutchups[0][1]*100))
    # logistic_mu_CV = int(round(logistic_mu_CV[0][1]*100))

    numeric_pred = 0
    oh_rf_pred = 0
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

    merged = (neural_pred3L3W[0][0]*0.5 + neural_predCross2[0][0]*0.5) * 100

    averaged = (neural_pred[0][0] + neural_pred2[0][0] + neural_pred3L3W[0][0] + neural_pred4L3W[0][0] +
                neural_pred4L4W[0][0]
                + neural_predCross[0][0] + neural_predCross2[0][0] + neural_predCross3[0][0] + neural_predCross4[0][
                    0]) / 9 * 100

    games = xin[-1]
    #strong_logistic, strong_logistic_CV = strong_logistic(games, logistic_input, xin, original_input)

    log =(s + "-" + str(int(round(numeric_pred)))
          + "%-" + str(int(round(old_numeric_pred)))
          + "%-" + str(int(round(oh_rf_pred)))+"%-"
          + "0%-"
          +  str(int(round(logistic_mu_CV)))
          + "%-" + str(int(round(logistic_mutchups)))+"%-"
          + "0%-"
          + "0%-"
          + "0%-"
          + "0%-"
          + "0%-"
          + "0%-"
          + str(int(round(merged))) +"%-"
          + str(int(round(averaged)))+"%"
          + "-" + str((int(round(neural_pred[0][0] * 100)))) + "%"
          + "-" + str((int(round(neural_pred2[0][0] * 100)))) + "%"
          + "-" + str((int(round(neural_pred3L3W[0][0] * 100))))+"%"
          + "-" + str((int(round(neural_pred4L3W[0][0] * 100))))+"%"
          + "-" + str((int(round(neural_pred4L4W[0][0] * 100))))+"%"
          + "-0%"
          + "-" + str((int(round(neural_predCross[0][0] * 100)))) + "%"
          + "-" + str((int(round(neural_predCross2[0][0] * 100)))) + "%"
          + "-" + str((int(round(neural_predCross3[0][0] * 100)))) + "%"
          + "-" + str((int(round(neural_predCross4[0][0] * 100)))) + "%"
          + "-0%-"
          + str(int(round(neural_meta[0] * 100))) + "%-"
          + str(int(round(sigma[0] * 100))) + "%-"
          + str(int(round(enhanced[0][0] * 100))) + "%"
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

coeffs_total = 4 * [float(0)]
coeffs_total = np.array(coeffs_total)
for i in range(len(input)):

    xin = input[i]

    coeffs = predict(input, or_input, y, i)
    coeffs_total += coeffs
    print(coeffs)
#
# print("coeffs total")
# print(coeffs_total/len(input))



