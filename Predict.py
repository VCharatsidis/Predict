f = open("Grubb.txt", "r")
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import copy
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import PredictV2
import logistic_mutchups
import rf_winrates
import rf_transformed
import preprocessed_logreg
import torch
from torch.autograd import Variable

def parse_one_hot(xin):

    data = ""
    if xin[0] ==1:
        data += "Hum "
    elif xin[1]==1:
        data += " Ne "
    elif xin[2]==1:
        data += " Orc "
    elif xin[3]==1:
        data += " Ra "
    elif xin[4]==1:
        data += " Ud"

    data += " vs "

    if xin[7]==1:
        data += " Hum "
    elif xin[8]==1:
        data += " Ne "
    elif xin[9]==1:
        data += " Orc "
    elif xin[10]==1:
        data += " Ra "
    elif xin[11]==1:
        data += " Ud"

    # amazonia = 0
    # concealed = 1
    # echo = 2
    # northren = 3
    # refuge = 4
    # swamped = 5
    # terenas = 6
    # turtle = 7
    # twisted = 8

    if len(xin) > 11:
        if xin[12] == 1:
            data += " amazonia "
        elif xin[13] == 1:
            data += " concealed "
        elif xin[14] == 1:
            data += " echo "
        elif xin[15] == 1:
            data += " northren "
        elif xin[16] == 1:
            data += " refuge"
        elif xin[17] == 1:
            data += " swamped "
        elif xin[18] == 1:
            data += " terenas "
        elif xin[19] == 1:
            data += " turtle "
        elif xin[20] == 1:
            data += " twisted"

        if xin[5] == 0:
            data += " tryhard "
        else:
            data += " request"

        data += " "+str(xin[21])+"%"

        data += " se "+str(xin[22])+" games"

    else:
        if xin[10] == 1:
            data += " tryhard "
        else:
            data += " request"

        data += " " + str(xin[11]) + "%"

        data += " se " + str(xin[12]) + " games"

    return data

def parse_x(xin, result):
    data = str(result)+"-"


    if xin[0] == 0:
        data += "Hum-"
    elif xin[0] == 1:
        data += "Ne-"
    elif xin[0] == 2:
        data += "Orc-"
    elif xin[0] == 3:
        data += "Ra-"
    elif xin[0] == 4:
        data += "Ud-"

    if xin[1] == 1:
        data += "t-"
    else:
        data += "r-"

    if xin[2] == 0:
        data += "Hum-"
    elif xin[2] == 1:
        data += "Ne-"
    elif xin[2] == 2:
        data += "Orc-"
    elif xin[2] == 3:
        data += "Ra-"
    elif xin[2] == 4:
        data += "Ud-"

    data += str(xin[3])
    data += "-"+str(xin[4])+"-"

    # amazonia = 0
    # concealed = 1
    # echo = 2
    # northren = 3
    # refuge = 4
    # swamped = 5
    # terenas = 6
    # turtle = 7
    # twisted = 8
    if(len(xin)>5):
        if xin[5] == 0:
            data += "amazonia"
        elif xin[5] == 1:
            data += "concealed"
        elif xin[5] == 2:
            data += "echo"
        elif xin[5] == 3:
            data += "northren"
        elif xin[5] == 4:
            data += "refuge"
        elif xin[5] == 5:
            data += "swamped"
        elif xin[5] == 6:
            data += "terenas"
        elif xin[5] == 7:
            data += "turtle"
        elif xin[5] == 8:
            data += "twisted"

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
    if X[4] < 58:
        continue

    X[5] = int(X[5])

    if X[5] < 15:
        X[4] = int(X[4] * 0.88)
    elif X[5] < 30:
        X[4] = int(X[4] * 0.93)

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
# scaller = StandardScaler()
#
# print("shape")
# print(input[:,3].shape)
#
# x3= input[:,3].reshape(-1,1)
# x3= scaller.fit_transform(x3)
#
# x4= input[:,4].reshape(-1,1)
# x4= scaller.fit_transform(x4)
#
# x3.reshape((len(input),))
# x4.reshape((len(input),))
#
# print(x3.shape)
# input[:,3] = x3[:,0]
# input[:,4] = x4[:,0]

input_h2o = copy.deepcopy(input)


input[:, 0] = labelencoder.fit_transform(input[:, 0])
input[:, 0] = [int(x) for x in input[:, 0]]

input[:, 1] = labelencoder.fit_transform(input[:, 1])
input[:, 1] = [int(x) for x in input[:, 1]]

input[:, 2] = labelencoder.fit_transform(input[:, 2])
input[:, 2] = [int(x) for x in input[:, 2]]

input[:, 5] = labelencoder.fit_transform(input[:, 5])
input[:, 5] = [int(x) for x in input[:, 5]]


input2 = copy.deepcopy(input)

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Estimators ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
errors = []
estimators = 500

classifier = RandomForestClassifier(n_estimators=estimators, random_state=0, oob_score=True)
classifier.fit(input, y)
oob_error1 = 1 - classifier.oob_score_
errors.append(oob_error1)
importances1 = classifier.feature_importances_
# amazonia = 0       swamped = 5
# concealed = 1      terenas = 6
# echo = 2           turtle = 7
# northren = 3       twisted = 8
# refuge = 4

# --------------------------------------------- Input -----------------------------------------------------------

#0-Hum-t-Hum-88-41-echo

xin = [4, 1, 1, 76, 1500, 4]
my_prediction = 38
Vagelis = 37
result = 0

write = True

rf_winrates_res, combo_importances = rf_winrates.predict(xin)
rf_trasformed, _ = rf_transformed.predict(xin)
logistic_mutchups = logistic_mutchups.logistic_reg(xin)
preprocessed_logreg_no_formula = preprocessed_logreg.logistic_reg(xin, False)[0][1]
preprocessed_logreg_formula = preprocessed_logreg.logistic_reg(xin, True)[0][1]





# Hum = 0
# Ne = 1
# Orc = 2
# Ra = 3
# Ud = 4


# xin3 = xin[3]
# xin3 = scaller.transform([[xin3]])
#
# xin4 = xin[4]
# xin4 = scaller.transform([[xin4]])
#
# xin[3] = xin3[0][0]
# xin[4] = xin4[0][0]
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

letter = [0 for _ in range(9)]
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
y_pred1 = classifier.predict_proba([xin])


# One Hot ############################################################

# amazonia = 0
# concealed = 1
# echo = 2
# northren = 3
# refuge = 4
# swamped = 5
# terenas = 6
# turtle = 7
# twisted = 8

#xin = [1., 0., 0., 0., 0.,      0., 1., 0., 0., 0.,   0., 0., 0., 1., 0., 0., 0., 0., 0.,       1,      89, 430]

# Hum = 0
# Ne = 1
# Orc = 2
# Ra = 3
# Ud = 4

onehotencoder = OneHotEncoder(categorical_features=[0, 1, 2, 5])
onehot_input = onehotencoder.fit_transform(input2).toarray()
logistic_input = copy.deepcopy(onehot_input)


X_train = onehot_input

mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

onehot_neural = onehot_encoded
onehot_neural = onehot_neural - mean
onehot_neural = onehot_neural / std
x = Variable(torch.FloatTensor([onehot_neural]))

model = torch.load('grubbyStar.model')
model.eval()
pred = model.forward(x)
print("neural prediction")
print(pred)
predi = pred
neural_pred = predi.detach().numpy()
print(neural_pred)


model2 = torch.load('grubbyStar2.model')
model2.eval()
pred2 = model2.forward(x)
print("neural prediction")
print(pred2)
predi2 = pred2
neural_pred2 = predi2.detach().numpy()
print(neural_pred2)


model5n = torch.load('grubbyStar5n.model')
model5n.eval()
pred5n = model5n.forward(x)
print("neural prediction")
print(pred5n)
predi5n = pred5n
neural_pred5n = predi5n.detach().numpy()
print(neural_pred5n)


estimators2 = 300
classifier2 = RandomForestClassifier(n_estimators=estimators2, random_state=0, oob_score=True)
classifier2.fit(onehot_input, y)
oob_error3 = 1 - classifier2.oob_score_
errors.append(oob_error3)

xin = onehot_encoded
y_pred3 = classifier2.predict_proba([xin])


# xin2 = [xin[0], xin[1], xin[2], xin[3], xin[4], xin[5], xin[6], xin[7], xin[8], xin[9], xin[10], xin[11], xin[21], xin[22]]
#
# input2 = input2[:, :-1]
# onehotencoder = OneHotEncoder(categorical_features=[0, 1, 2])
# onehot_input = onehotencoder.fit_transform(input2).toarray()
#
#
# classifier3 = RandomForestClassifier(n_estimators=estimators2, random_state=0, oob_score=True)
# classifier3.fit(onehot_input, y)
# oob_error4 = 1 - classifier3.oob_score_
# errors.append(oob_error4)
# y_pred4 = classifier3.predict_proba([xin2])

################################## Logistic  -------------------------------------------------


# plt.plot(logistic_input[:, -1], logistic_input[:, -2], 'ro')
# plt.ylabel('stats')
# plt.xlabel('games')
# plt.show()


clf = LogisticRegression(solver='lbfgs', max_iter=400, class_weight='balanced').fit(logistic_input, y)
y_pred_logistic = clf.predict_proba([xin])

# show = [a[-1] for a in logistic_input if a[-1] <= 15]
# show2 = [a[-2] for a in logistic_input if a[-1] <= 15]
# plt.plot(show, show2, 'ro')
# plt.ylabel('stats')
# plt.xlabel('games')
# plt.show()

games = xin[-1]
strong_logistic = 0

if games <= 15:
    logistic_input15 = [a[:-1] for a in logistic_input if a[-1] <= 20]
    y_15 = [a[0] for a in original_input if int(a[-2]) <= 20]
    clf15 = LogisticRegression(solver='lbfgs', max_iter=1000, class_weight='balanced').fit(logistic_input15, y_15)
    y_pred_logistic15 = clf15.predict_proba([xin[:-1]])
    strong_logistic = y_pred_logistic15[0][1]

elif games < 40:
    logistic_input35 = [a[:-1] for a in logistic_input if 16 <= a[-1] <= 60]
    y_35 = [a[0] for a in original_input if 16 <= int(a[-2]) <= 60]
    clf35 = LogisticRegression(solver='lbfgs', max_iter=1000, class_weight='balanced').fit(logistic_input35, y_35)
    print(len(y_35))
    y_pred_logistic35 = clf35.predict_proba([xin[:-1]])
    strong_logistic = y_pred_logistic35[0][1]

else:
    logistic_inputRest = [a[:-1] for a in logistic_input if 40 <= a[-1]]
    y_Rest = [a[0] for a in original_input if 40 <= int(a[-2])]
    clfRest = LogisticRegression(solver='lbfgs', max_iter=500, class_weight='balanced').fit(logistic_inputRest, y_Rest)
    print(len(y_Rest))

    y_pred_logisticRest = clfRest.predict_proba([xin[:-1]])
    strong_logistic = y_pred_logisticRest[0][1]


#print(y_pred_logisticRest)


classifier = RandomForestClassifier(n_estimators=estimators, random_state=0, oob_score=True)
classifier.fit(input, y)


###############################  K nearest neightours   ##############################################


print(parse_one_hot(xin))

np.set_printoptions(precision=2)
importances = classifier.feature_importances_
importances2 = classifier2.feature_importances_
importances2 = ['%.2f'%(float(a)) for a in importances2]


s = parse_x(to_print, result)

pred1 = int(round(y_pred1[0][1]*100))
pred3 = int(round(y_pred3[0][1]*100))

logistic_pred = int(round(y_pred_logistic[0][1]*100))
logistic_mutchups = int(round(logistic_mutchups[0][1]*100))


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ RESULT #############################################################################
print("")
if games < 40:
    preprocessed_logreg_no_formula = 0
    preprocessed_logreg_formula = 0
    logistic_mutchups = 0

log =(s +"-" + str(pred1)
      +"%-" + str(0)
      +"%-" + str(pred3)
      +"%-" + str(0)
      +"%-" + str(rf_winrates_res)
      +"%-" + str(logistic_mutchups)
      +"%-" + str(logistic_pred)
      +"%-" + str(int(round(strong_logistic * 100)))
      +"%-" + str(rf_trasformed)
      +"%-" + str(Vagelis)
      +"%-" + str(my_prediction)+"%"
      +"-0%-"
      + str(int(round(preprocessed_logreg_no_formula * 100))) + "%-"
      + str(int(round(preprocessed_logreg_formula*100))) +"%"
      +"-"+ str(str(int(round(neural_pred[0][0]*100)))) +"%"
      +"-"+ str(str(int(round(neural_pred2[0][0]*100)))) +"%-"
      +"-"+ str(str(int(round(neural_pred5n[0][0]*100)))) +"%-"
      +"\n")

print(log)

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ RESULT #############################################################################
print("avg opponent winrate: " + str(avg_opponents_winrate))
print("observed Grubby winrates: " + str(observed_grubby_winrates))
print("combo importances")
print(combo_importances)

print("strong logistic: " + str(int(round(strong_logistic * 100))) + "%")
print("neural pred: "+ str(int(round(neural_pred[0][0]*100))) + "%")
print("neural pred2: "+ str(int(round(neural_pred2[0][0]*100))) + "%")
print("neural pred5n: "+ str(int(round(neural_pred5n[0][0]*100))) + "%")
print("preprocessed log_reg formula: " + str(int(round(preprocessed_logreg_formula * 100))) + "%")
print("preprocessed log_reg : " + str(int(round(preprocessed_logreg_no_formula * 100))) + "%")
print("random forests formula winrates: " + str(rf_trasformed)+"%")
print("random forests winrates: " + str(rf_winrates_res) + "%")
print("matchups logistic: " + str(logistic_mutchups)+"%")
print("normal logistic: " + str(logistic_pred)+"%")
print("one hot rf: " + str(pred3) + "%")




#errors = np.array(errors)
#print(str(errors)+"--"+str(estimators))


#################################       H2O       #############################################


# import h2o
# from h2o.estimators.random_forest import H2ORandomForestEstimator
#
# h2o.init()
#
# # get training and prediction data sets
# air = input_h2o
# # only use columns 1 through 11
#
#
# #subset the training data into train and validation sets
# r = input_h2o[0].runif()
# air_train = input_h2o[r < 0.8]
# air_valid = input_h2o[r >= 0.8]
#
# # specify your features and response column
# myX = ["GrubbyRace", "Effort", "OpponentRace", "Stats", "NumberGames", "Map"]
# myY = "Won"
#
# # build and train your model
# rf_bal = H2ORandomForestEstimator(seed=12, ntrees=150, balance_classes=True)
# rf_bal.train(x=input_h2o, y=y, training_frame=air_train, validation_frame=air_valid)
#
#
# # show predicted yes/no output with probability for yes and no
# rf_bal.predict(input_h2o[1:6])

################################## Score between classifiers ################################

if write:
    file = open("Grubb.txt", "a")
    file.write(s+"\n")
    file.close()
    #
    file = open("automagic.txt", "a")
    file.write(log)
    file.close()