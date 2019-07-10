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

import config


f = open("../logs/Grubb.txt", "r")

def parse_x(xin, result):

    data = str(result)+"-"

    data += config.races[xin[0]]+"-"
    data += config.tryhard[xin[1]]
    data += config.races[xin[2]]+"-"
    data += str(xin[3])
    data += "-"+str(xin[4])+"-"

    if(len(xin)>5):
        data += config.maps[xin[5]]

    return data

contents = f.readlines()


counter = 0
avg_opponents_winrate = 0
observed_grubby_wins = {'Hum': 0, 'Ne': 0, 'Orc': 0, 'Ra': 0, 'Ud': 0}
observer_grubby_games = {'Hum': 0, 'Ne': 0, 'Orc': 0, 'Ra': 0, 'Ud': 0}
observed_grubby_winrates = {'Hum': 0, 'Ne': 0, 'Orc': 0, 'Ra': 0, 'Ud': 0}

wins_less_than_60 = 0
games_less_than_60 = 0


def predict(i):
    input_cp = []
    counter = 0
    for l in contents:
        X = l.split('-')

        X[4] = int(X[4])
        if X[4] < 55:
            counter += 1
            continue

        X[5] = int(X[5])

        if X[5] < 15:
            X[4] = int(X[4] * 0.88)
        elif X[5] < 30:
            X[4] = int(X[4] * 0.93)

        X[6] = X[6].rstrip("\n")

        X = np.array(X)
        input_cp.append(X)

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


    xin = copy.deepcopy(input_cp[i])
    xin = [int(x) for x in xin]

    print(xin)
    print(len(input_cp))
    original_input_for_strong_log_reg = np.delete(original_input_for_strong_log_reg, i, axis=0)
    input_cp = np.delete(input_cp, i, axis=0)
    print(len(input_cp))

    result = y[i]
    y = np.delete(y, i, axis=0)

    original_input = copy.deepcopy(input_cp)

    labelencoder = LabelEncoder()


    input2 = copy.deepcopy(input_cp)

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Estimators ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    errors = []

    old_numeric_rf = RandomForestClassifier(n_estimators=2000, min_samples_split=10, oob_score=True)
    old_numeric_rf.fit(input_cp, y)

    numeric_rf = RandomForestClassifier(n_estimators=2000,
                                        random_state=0,
                                        oob_score=True,
                                        min_samples_leaf=18,
                                        min_samples_split=5,
                                        max_features=3,
                                        max_depth=20,
                                        bootstrap=True)
    numeric_rf.fit(input_cp, y)
    oob_error1 = 1 - numeric_rf.oob_score_
    errors.append(oob_error1)
    importances1 = numeric_rf.feature_importances_



    np.set_printoptions(precision=2)
    importances1 = ['%.2f'%(float(a)) for a in importances1]


    logistic_mutchups, logistic_mu_CV = logistic_reg(xin, "../logs/Grubb.txt", i)


    # Hum = 0
    # Ne = 1
    # Orc = 2
    # Ra = 3
    # Ud = 4

    to_print = copy.deepcopy(xin)
    print(parse_x(to_print, result))


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

    # 22222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222

    y_pred1 = numeric_rf.predict_proba([xin])
    old_numeric_pred = old_numeric_rf.predict_proba([xin])


    # One Hot ############################################################


    onehotencoder = OneHotEncoder(categorical_features=[0, 1, 2, 5])
    onehot_input = onehotencoder.fit_transform(input2).toarray()
    logistic_input = copy.deepcopy(onehot_input)
    #
    #
    # ####################################################   Neural nets   #####################################
    #
    #
    neural_pred, neural_pred2, neural_pred3L3W, neural_pred4L3W, neural_pred4L4W,\
    neural_predTest, neural_predCross, neural_predCross2, neural_predCross3,\
    neural_predCross4 = load_models(onehot_encoded)
    #
    #
    # ############################################################## one hot rf ###############################
    #
    one_hot_rf = RandomForestClassifier(n_estimators=2000,
                                        random_state=0,
                                        oob_score=True,
                                        min_samples_leaf=15,
                                        min_samples_split=29,
                                        max_features=21,
                                        max_depth=20,
                                        bootstrap=True)

    one_hot_rf.fit(onehot_input, y)
    oob_error3 = 1 - one_hot_rf.oob_score_
    errors.append(oob_error3)

    xin = onehot_encoded
    y_pred3 = one_hot_rf.predict_proba([xin])

    importances2 = one_hot_rf.feature_importances_


    ################################## Logistic  -------------------------------------------------


    clf = LogisticRegression(solver='lbfgs', max_iter=400).fit(logistic_input, y)
    y_pred_logistic = clf.predict_proba([xin])

    print("xin before strong log reg "+str(xin))
    print("original_input_for_strong_log_reg[0] before strong log reg: " + str(original_input_for_strong_log_reg[0]))

    games = xin[-1]

    strong_logistic = 0
    strong_logistic_CV = 0

    if games <= 15:
        logistic_input15 = [a[:-1] for a in logistic_input if a[-1] <= 20]
        y_15 = [a[0] for a in original_input_for_strong_log_reg if int(a[-2]) <= 20]
        clf15 = LogisticRegression(solver='lbfgs', max_iter=1000).fit(logistic_input15, y_15)
        clf15CV = LogisticRegressionCV(solver='lbfgs', max_iter=1000, cv=10).fit(logistic_input15, y_15)

        y_pred_logistic15_CV = clf15CV.predict_proba([xin[:-1]])
        y_pred_logistic15 = clf15.predict_proba([xin[:-1]])

        strong_logistic = y_pred_logistic15[0][1]
        strong_logistic_CV = y_pred_logistic15_CV[0][1]

    elif games < 40:
        logistic_input35 = [a[:-1] for a in logistic_input if 16 <= a[-1] <= 60]
        y_35 = [a[0] for a in original_input_for_strong_log_reg if 16 <= int(a[-2]) <= 60]
        clf35 = LogisticRegression(solver='lbfgs', max_iter=1000).fit(logistic_input35, y_35)
        clf35CV = LogisticRegressionCV(solver='lbfgs', max_iter=1000, cv=10).fit(logistic_input35, y_35)

        y_pred_logistic15_CV = clf35CV.predict_proba([xin[:-1]])
        y_pred_logistic35 = clf35.predict_proba([xin[:-1]])

        strong_logistic = y_pred_logistic35[0][1]
        strong_logistic_CV = y_pred_logistic15_CV[0][1]

    else:
        logistic_inputRest = [a[:-1] for a in logistic_input if 40 <= a[-1]]
        y_Rest = [a[0] for a in original_input_for_strong_log_reg if 40 <= int(a[-2])]

        clfRest = LogisticRegression(solver='lbfgs', max_iter=1000).fit(logistic_inputRest, y_Rest)
        clfRest_CV = LogisticRegressionCV(solver='lbfgs', max_iter=1000, cv=10).fit(logistic_inputRest, y_Rest)

        y_pred_logisticRest = clfRest.predict_proba([xin[:-1]])
        y_pred_logisticRestCV = clfRest_CV.predict_proba([xin[:-1]])

        strong_logistic = y_pred_logisticRest[0][1]
        strong_logistic_CV = y_pred_logisticRestCV[0][1]
    #
    # ###############################  K nearest neightours   ##############################################
    #
    #
    s = parse_x(to_print, result)
    #
    old_numeric_pred = int(round(old_numeric_pred[0][1]*100))
    pred1 = int(round(y_pred1[0][1]*100))
    pred3 = int(round(y_pred3[0][1]*100))
    logistic_pred = int(round(y_pred_logistic[0][1]*100))
    logistic_mutchups = int(round(logistic_mutchups[0][1]*100))
    logistic_mu_CV = int(round(logistic_mu_CV[0][1]*100))
    strong_logistic = int(round(strong_logistic*100))
    strong_logistic_CV = int(round(strong_logistic_CV*100))

    # old_numeric_pred = 0
    # pred1 = 0
    # pred3 = 0
    # logistic_pred = 0
    # logistic_mutchups = 0
    # logistic_mu_CV = 0
    # strong_logistic = 0
    # strong_logistic_CV = 0

    # neural_pred[0][0] = 0
    # neural_pred2[0][0] = 0
    # neural_pred3L3W[0][0] = 0
    # neural_pred4L3W[0][0] = 0
    # neural_pred4L4W[0][0] = 0
    # neural_predTest[0][0] = 0
    #
    # neural_predCross2[0][0] = 0
    # neural_predCross3[0][0] = 0


    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ RESULT #############################################################################
    print("")
    if games < 30:
        logistic_mutchups = 0
        logistic_mu_CV = 0

    avg_neural = 0
    avg_neural = (int(round(neural_pred[0][0] * 100)) +
                  int(round(neural_pred2[0][0] * 100)) +
                  int(round(neural_pred3L3W[0][0] * 100)) +
                  int(round(neural_pred4L3W[0][0] * 100)) +
                  int(round(neural_pred4L4W[0][0] * 100)) +
                  int(round(neural_predTest[0][0] * 100)))/6

    log =(s + "-" + str(pred1)
          + "%-" + str(old_numeric_pred)
          + "%-" + str(pred3)
          + "%-" + str(strong_logistic_CV)
          + "%-" + str(logistic_mu_CV)
          + "%-" + str(logistic_mutchups)
          + "%-" + str(logistic_pred)
          + "%-" + str(strong_logistic)+"%-"
          + "0%-"
          + "0%-"
          + "0%-"
          + "0%-"
          + "0%-"
          + "0%"
          +"-" + str((int(round(neural_pred[0][0]*100)))) + "%"
          +"-" + str((int(round(neural_pred2[0][0]*100)))) + "%"
          +"-" + str((int(round(neural_pred3L3W[0][0] * 100))))+"%"
          +"-" + str((int(round(neural_pred4L3W[0][0] * 100))))+"%"
          +"-" + str((int(round(neural_pred4L4W[0][0] * 100))))+"%"
          +"-" + str(round(avg_neural))+"%"
          +"-" + str((int(round(neural_predCross[0][0]*100)))) + "%"
          + "-" + str((int(round(neural_predCross2[0][0] * 100)))) + "%"
          + "-" + str((int(round(neural_predCross3[0][0] * 100)))) + "%"
          + "-" + str((int(round(neural_predCross4[0][0] * 100)))) + "%"
          +"\n")

    print(log)


################################## Score between classifiers ################################

    file = open("../logs/new_predictions.txt", "a")
    file.write(log)
    file.close()


input = []

for l in contents:

    X = l.split('-')

    X[4] = int(X[4])
    if X[4] < 55:
        continue

    X[5] = int(X[5])

    if X[5] < 15:
        X[4] = int(X[4] * 0.88)
    elif X[5] < 30:
        X[4] = int(X[4] * 0.93)

    X[6] = X[6].rstrip("\n")

    X = np.array(X)
    input.append(X)

input = np.array(input)

for i in range(len(input)):

    xin = input[i]

    predict( i)




