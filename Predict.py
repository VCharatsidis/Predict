import numpy as np
from sklearn.ensemble import RandomForestClassifier
import copy
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from logistic_reggresions import logistic_mutchups, strong_logistic
from load_models import load_models

import config


f = open("logs/Grubb.txt", "r")

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

input = []
counter = 0
avg_opponents_winrate = 0
observed_grubby_wins = {'Hum': 0, 'Ne': 0, 'Orc': 0, 'Ra': 0, 'Ud': 0}
observer_grubby_games = {'Hum': 0, 'Ne': 0, 'Orc': 0, 'Ra': 0, 'Ud': 0}
observed_grubby_winrates = {'Hum': 0, 'Ne': 0, 'Orc': 0, 'Ra': 0, 'Ud': 0}

wins_less_than_60 = 0
games_less_than_60 = 0

for l in contents:

    X = l.split('-')

    X[4] = int(X[4])
    if X[4] < 55:
        continue

    X[5] = int(X[5])

    X[6] = X[6].rstrip("\n")

    X = np.array(X)
    input.append(X)

    observed_grubby_wins[X[1]] += int(X[0])
    observer_grubby_games[X[1]] += 1

    avg_opponents_winrate += int(X[4])
    counter += 1


for key in observed_grubby_wins.keys():
    observed_grubby_winrates[key] = observed_grubby_wins[key] / observer_grubby_games[key]

avg_opponents_winrate /= counter

input = np.array(input)
original_input = copy.deepcopy(input)

labelencoder = LabelEncoder()

y = input[:, 0]
input = input[:, 1:]


input[:, 0] = labelencoder.fit_transform(input[:, 0])
input[:, 0] = [int(x) for x in input[:, 0]]

input[:, 1] = labelencoder.fit_transform(input[:, 1])
input[:, 1] = [int(x) for x in input[:, 1]]

input[:, 2] = labelencoder.fit_transform(input[:, 2])
input[:, 2] = [int(x) for x in input[:, 2]]

input[:, 5] = labelencoder.fit_transform(input[:, 5])
input[:, 5] = [int(x) for x in input[:, 5]]


print(config.maps[int(input[0, 0:6][5])])
print(config.maps[int(input[1, 0:6][5])])
print(config.maps[int(input[2, 0:6][5])])
print(config.maps[int(input[3, 0:6][5])])
print(config.maps[int(input[4, 0:6][5])])
print(config.maps[int(input[5, 0:6][5])])
print(config.maps[int(input[6, 0:6][5])])
print(config.maps[int(input[7, 0:6][5])])
print(config.maps[int(input[8, 0:6][5])])
print(config.maps[int(input[9, 0:6][5])])


input2 = copy.deepcopy(input)

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Estimators ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
errors = []

old_numeric_rf = RandomForestClassifier(n_estimators=1000, min_samples_split=10, oob_score=True)
old_numeric_rf.fit(input, y)
oob_error_old = 1 - old_numeric_rf.oob_score_
importances_old = old_numeric_rf.feature_importances_


# {'n_estimators': 2000, 'min_samples_split': 17, 'min_samples_leaf': 17, 'max_features': 6, 'max_depth': 25, 'bootstrap': True}
# {'n_estimators': 2000, 'min_samples_split': 5, 'min_samples_leaf': 18, 'max_features': 5, 'max_depth': 5}

numeric_rf = RandomForestClassifier(n_estimators=1000,
                                    random_state=0,
                                    oob_score=True,
                                    min_samples_leaf=18,
                                    min_samples_split=5,
                                    max_features=3,
                                    max_depth=20,
                                    bootstrap=True)
numeric_rf.fit(input, y)
oob_error1 = 1 - numeric_rf.oob_score_
errors.append(oob_error1)
importances1 = numeric_rf.feature_importances_



np.set_printoptions(precision=2)
importances1 = ['%.2f'%(float(a)) for a in importances1]



# amazonia = 0       swamped = 5      nomad = 10
# concealed = 1      terenas = 6
# echo = 2           turtle = 7
# northren = 3       twisted = 8
# refuge = 4         ancient = 9

# --------------------------------------------- Input -----------------------------------------------------------


xin = [2, 0, 0, 81, 42, 8]
my_prediction = 55
Vagelis = 0
result = 1

write = True
NEW_PATCH = 500

path = "logs/Grubb.txt"
logistic_mutchups, logistic_mu_CV = logistic_mutchups.logistic_reg(xin, path, 2000)


# Hum = 0
# Ne = 1
# Orc = 2
# Ra = 3
# Ud = 4



print(xin)

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


# print(parse_one_hot(xin))

# 22222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222
y_pred1 = numeric_rf.predict_proba([xin])
old_numeric_pred = old_numeric_rf.predict_proba([xin])



# One Hot ############################################################

# from sklearn.compose import ColumnTransformer
# # The last arg ([0]) is the list of columns you want to transform in this step
# ct = ColumnTransformer([("race grub", OneHotEncoder()),("tryhard", OneHotEncoder()), ("race opp", OneHotEncoder()), "map"])


onehotencoder = OneHotEncoder(categorical_features=[0, 1, 2, 5])
onehot_input = onehotencoder.fit_transform(input2).toarray()
logistic_input = copy.deepcopy(onehot_input)


####################################################   Neural nets   #####################################


neural_pred, neural_pred2, neural_pred3L3W, neural_pred4L3W, neural_pred4L4W,\
neural_predCross, neural_predCross2, neural_predCross3,\
neural_predCross4 = load_models(onehot_encoded)


############################################################## one hot rf ###############################


# {'n_estimators': 2000, 'min_samples_split': 29, 'min_samples_leaf': 15, 'max_features': 21, 'max_depth': 20, 'bootstrap': True}
# {'n_estimators': 2000, 'min_samples_split': 13, 'min_samples_leaf': 18, 'max_features': 22, 'max_depth': 22, 'bootstrap': True}
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
importances2 = ['%.2f'%(float(a)) for a in importances2]



################################## Logistic  -------------------------------------------------


# plt.plot(logistic_input[:, -1], logistic_input[:, -2], 'ro')
# plt.ylabel('stats')
# plt.xlabel('games')
# plt.show()


clf = LogisticRegression(solver='lbfgs', max_iter=400).fit(logistic_input, y)
y_pred_logistic = clf.predict_proba([xin])

# show = [a[-1] for a in logistic_input if a[-1] <= 15]
# show2 = [a[-2] for a in logistic_input if a[-1] <= 15]
# plt.plot(show, show2, 'ro')
# plt.ylabel('stats')
# plt.xlabel('games')
# plt.show()

games = xin[-1]
strong_logistic, strong_logistic_CV = strong_logistic.strong_logistic(games, logistic_input, xin, original_input)

###############################  K nearest neightours   ##############################################


s = parse_x(to_print, result)

old_numeric_pred = int(round(old_numeric_pred[0][1]*100))
pred1 = int(round(y_pred1[0][1]*100))
pred3 = int(round(y_pred3[0][1]*100))
logistic_pred = int(round(y_pred_logistic[0][1]*100))
logistic_mutchups = int(round(logistic_mutchups[0][1]*100))
logistic_mu_CV = int(round(logistic_mu_CV[0][1]*100))
strong_logistic = int(round(strong_logistic*100))
strong_logistic_CV = int(round(strong_logistic_CV*100))


# if neural_predCross4[0][0] > 0.66:
#     m1 = neural_predCross4[0][0]
# else:
#     m1 = neural_pred2[0][0]
#
# # merged = (neural_predCross4[0][0] + neural_pred2[0][0])/2
#
# merged = (neural_predCross4[0][0] * 0.4 + m1 * 0.6) * 100


if neural_predCross4[0][0] > 0.66:
    m2 = neural_predCross4[0][0] * 98
else:
    m2 = neural_pred2[0][0] * 102

# one hot rf in predictTool
merged = m2 * 0.4 + neural_predCross[0][0] * 100 * 0.6


if merged < 0.5:
    result = 0

# logistic mu CV in predicTool
averaged = neural_pred[0][0] * 100 * 0.6 + neural_predCross4[0][0] * 99 * 0.4

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ RESULT #############################################################################
print("")
if games < 30:
    logistic_mutchups = 0
    logistic_mu_CV = 0


log =(s + "-" + str(pred1)
      + "%-" + str(old_numeric_pred)
      + "%-" + str(pred3)
      + "%-" + str(strong_logistic_CV)
      + "%-" + str(logistic_mu_CV)
      + "%-" + str(logistic_mutchups)
      + "%-" + str(logistic_pred)
      + "%-" + str(strong_logistic)+"%-"
      + "0%-"
      + str(Vagelis)
      +"%-" + str(my_prediction) + "%"
      +"-0%-"
      + str(int(round(merged))) + "%-"
      + str(int(round(averaged))) + "%"
      +"-" + str((int(round(neural_pred[0][0]*100)))) + "%"
      +"-" + str((int(round(neural_pred2[0][0]*100)))) + "%"
      +"-" + str((int(round(neural_pred3L3W[0][0] * 100))))+"%"
      +"-" + str((int(round(neural_pred4L3W[0][0] * 100))))+"%"
      +"-" + str((int(round(neural_pred4L4W[0][0] * 100))))+"%"
      +"-0%"
      +"-" + str((int(round(neural_predCross[0][0]*100)))) + "%"
      + "-" + str((int(round(neural_predCross2[0][0] * 100)))) + "%"
      + "-" + str((int(round(neural_predCross3[0][0] * 100)))) + "%"
      + "-" + str((int(round(neural_predCross4[0][0] * 100)))) + "%"
      +"\n")

print(log)

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ RESULT #############################################################################
print("avg opponent winrate: " + str(avg_opponents_winrate))
print("observed Grubby winrates: " + str(observed_grubby_winrates))

print("numeric rf: " + str(pred1) + "%")
print("old numeric rf: " + str(old_numeric_pred) + "%")
print("matchups logistic: " + str(logistic_mutchups)+"%")
print("matchups logistic CV: " + str(logistic_mu_CV)+"%")
print("strong logistic: " + str(strong_logistic) + "%")
print("strong logistic CV: " + str(strong_logistic_CV) + "%")
print("normal logistic: " + str(logistic_pred)+"%")
print("one hot rf: " + str(pred3) + "%")
print("")
print("merged: " + str(int(round(merged))) + "%")
print("logistic_mu_CV: " + str(int(round(averaged))) + "%")
print("neural Cross: " + str(int(round(neural_predCross[0][0]*100))) + "%")
print("neural Cross 2: " + str(int(round(neural_predCross2[0][0]*100))) + "%")
print("neural Cross 3: " + str(int(round(neural_predCross3[0][0]*100))) + "%")
print("neural Cross 4: " + str(int(round(neural_predCross4[0][0]*100))) + "%")
print("neural pred: " + str(int(round(neural_pred[0][0]*100))) + "%")
print("neural pred2: " + str(int(round(neural_pred2[0][0]*100))) + "%")
print("neural pred3L-3W: " + str(int(round(neural_pred3L3W[0][0] * 100))) + "%")
print("neural pred4L-3W: " + str(int(round(neural_pred4L3W[0][0] * 100))) + "%")
print("neural pred4L4W: " + str(int(round(neural_pred4L4W[0][0] * 100)))+"%")


print()
print("oob error numeric rf: " + str(oob_error1))
print("importances numeric rf: " + str(importances1))
print()
print("oob error old numeric rf: " + str(oob_error_old))
print("old importances " + str(importances_old))
print()
print("oob error one hot rf: " + str(oob_error3))
print("one hot importances: " + str(importances2))




#################################       H2O       #############################################

# import h2o
# import os
# from h2o import H2OFrame
#
#
# h2o.init()
# h2o.remove_all()
# from h2o.estimators.random_forest import H2ORandomForestEstimator
# covtype_df = h2o.import_file(os.path.realpath("GrubbC.csv"))
#
# train, valid, test = covtype_df.split_frame([0.8, 0.1], seed=1234)
# covtype_X = covtype_df.col_names[1:]     #last column is Cover_Type, our desired response variable
# covtype_y = covtype_df.col_names[0]
#
# print(test)
# rf_v1 = H2ORandomForestEstimator(
#     model_id="rf_covType_v1",
#     categorical_encoding='auto',
#     ntrees=500,
#     stopping_rounds=2,
#     score_each_iteration=True,
#     seed=1000000)
#
# rf_v1.train(covtype_X, covtype_y, training_frame=train)
# rf_v1.score_history()
#
# new = H2OFrame(z)
# h2o_res = rf_v1.predict(test_data=new)
#
# print("h2o " + str(h2o_res))
# h2o.shutdown(prompt=False)


################################## Score between classifiers ################################

if write:
    file = open("logs/Grubb.txt", "a")
    file.write(s+"\n")
    file.close()
    #
    file = open("logs/automagic.txt", "a")
    file.write(log)
    file.close()
