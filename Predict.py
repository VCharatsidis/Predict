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
import combolearning
import predictTransformed

def parse_one_hot(xin):
    print(xin)
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
    print("len xin "+str(len(xin)))
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

def parse_x(xin):
    data = "1-"


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

def less_buckets(games):
    bucket = 0

    if games <= 20:
        bucket = 1
    elif games <= 60:
        bucket = 2
    elif games <= 150:
        bucket = 3
    else:
        bucket = 4

    return bucket

def bucket_games(games):
    bucket = 0

    if games <= 12:
        bucket = 12
    elif games <= 20:
        bucket = 20
    elif games <= 35:
        bucket = 35
    elif games <= 60:
        bucket = 60
    elif games <= 120:
        bucket = 120
    else:
        bucket = 1000

    return bucket

contents = f.readlines()

input = []
counter = 0
for l in contents:

    X = l.split('-')
    # z = int(X[4])
    # stats_adipalou = (round(z/5))*5
    # X[4] = stats_adipalou
    # games_adipalou = int(X[5])
    # b = less_buckets(games_adipalou)
    # X[5] = b
    # X[6] = X[6].rstrip("\n")

    X[4] = int(X[4])
    X[5] = int(X[5])
    X[6] = X[6].rstrip("\n")

    X = np.array(X)
    input.append(X)

    # print("bucket "+str(b))
    # print("stats adipalou rounded "+str(stats_adipalou))
    # print("stats adipalou "+str(z))

    print(X)
    print(l)
    counter += 1


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
print("scaller-----------------------------------------------------------------------------")
print(input[-1])
input_h2o = copy.deepcopy(input)

print(y)
print(input)
print(input[18])
#
# input = labelencoder.fit_transform(input)
# print(input)
input[:, 0] = labelencoder.fit_transform(input[:, 0])
input[:, 0] = [int(x) for x in input[:, 0]]

input[:, 1] = labelencoder.fit_transform(input[:, 1])
input[:, 1] = [int(x) for x in input[:, 1]]

input[:, 2] = labelencoder.fit_transform(input[:, 2])
input[:, 2] = [int(x) for x in input[:, 2]]

input[:, 5] = labelencoder.fit_transform(input[:, 5])
input[:, 5] = [int(x) for x in input[:, 5]]
print(input)

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

xin = [2, 1, 4, 60, 1200, 0]

predComboLeanring, combo_importances = combolearning.predict(xin)
rf_trasformed, _ = predictTransformed.predict(xin)
predV2_logistic = PredictV2.logistic_reg(xin)
logistic_mutchups, logit_mu = logistic_mutchups.logistic_reg(xin)

write = False
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
print("hsasasasasasasasasasasasasasasasasasasasasasasasasasasasasasa")

to_print = copy.deepcopy(xin)
print(parse_x(to_print))


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


print("manual one hot "+str(onehot_encoded))
print(len(onehot_encoded))
# print(parse_one_hot(xin))

# 22222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222
y_pred1 = classifier.predict_proba([xin])
print(xin)
print("Chanses that Grubby wins "+str(y_pred1[0][1]*100)+"%")

#   Without map prediction   ##############################################

print("")
print("Without map prediciton ")
input = input[:, :-1]
classifier.fit(input, y)
oob_error2 = 1 - classifier.oob_score_
errors.append(oob_error2)
xin = xin[:-1]
y_pred2 = classifier.predict_proba([xin])


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

estimators2 = 300
classifier2 = RandomForestClassifier(n_estimators=estimators2, random_state=0, oob_score=True)
classifier2.fit(onehot_input, y)
oob_error3 = 1 - classifier2.oob_score_
errors.append(oob_error3)

xin = onehot_encoded
y_pred3 = classifier2.predict_proba([xin])


xin2 = [xin[0], xin[1], xin[2], xin[3], xin[4], xin[5], xin[6], xin[7], xin[8], xin[9], xin[10], xin[11], xin[21], xin[22]]

input2 = input2[:, :-1]
onehotencoder = OneHotEncoder(categorical_features=[0, 1, 2])
onehot_input = onehotencoder.fit_transform(input2).toarray()

print("")
print("Without map prediciton ")

classifier3 = RandomForestClassifier(n_estimators=estimators2, random_state=0, oob_score=True)
classifier3.fit(onehot_input, y)
oob_error4 = 1 - classifier3.oob_score_
errors.append(oob_error4)

print(xin2)
y_pred4 = classifier3.predict_proba([xin2])

################################## Logistic  -------------------------------------------------


# plt.plot(logistic_input[:, -1], logistic_input[:, -2], 'ro')
# plt.ylabel('stats')
# plt.xlabel('games')
# plt.show()


clf = LogisticRegression(solver='lbfgs', max_iter=800).fit(logistic_input, y)
y_pred_logistic, logit1 = clf.predict_proba([xin])

# show = [a[-1] for a in logistic_input if a[-1] <= 15]
# show2 = [a[-2] for a in logistic_input if a[-1] <= 15]
# plt.plot(show, show2, 'ro')
# plt.ylabel('stats')
# plt.xlabel('games')
# plt.show()


logistic_input15 = [a[:-1] for a in logistic_input if a[-1] <= 20]
y_15 = [a[0] for a in original_input if int(a[-2]) <= 20]
clf15 = LogisticRegression(solver='lbfgs', max_iter=1000).fit(logistic_input15, y_15)
print(len(y_15))


logistic_input35 = [a[:-1] for a in logistic_input if 16 <= a[-1] <= 45]
y_35 = [a[0] for a in original_input if 16 <= int(a[-2]) <= 45]
clf35 = LogisticRegression(solver='lbfgs', max_iter=1000).fit(logistic_input35, y_35)
print(len(y_35))


logistic_input70 = [a[:-1] for a in logistic_input if 36 <= a[-1] <= 80]
y_70 = [a[0] for a in original_input if 36 <= int(a[-2]) <= 80]
clf70 = LogisticRegression(solver='lbfgs', max_iter=1000).fit(logistic_input70, y_70)
print(len(y_70))


logistic_inputRest = [a[:-1] for a in logistic_input if 70 <= a[-1]]
y_Rest = [a[0] for a in original_input if 70 <= int(a[-2])]
clfRest = LogisticRegression(solver='lbfgs', max_iter=500).fit(logistic_inputRest, y_Rest)
print(len(y_Rest))

games = xin[-1]
correct_logistic = 0

y_pred_logistic15, logit = clf15.predict_proba([xin[:-1]])
y_pred_logistic35, logit = clf35.predict_proba([xin[:-1]])
y_pred_logistic70, logit = clf70.predict_proba([xin[:-1]])
y_pred_logisticRest, logitRest = clfRest.predict_proba([xin[:-1]])

print(y_pred_logisticRest)
print(logit)

classifier = RandomForestClassifier(n_estimators=estimators, random_state=0, oob_score=True)
classifier.fit(input, y)

if games <= 15:
    correct_logistic = y_pred_logistic15[0][1]
elif games <= 35:
    correct_logistic = y_pred_logistic35[0][1]
elif games <= 70:
    correct_logistic = y_pred_logistic70[0][1]
else:
    correct_logistic = y_pred_logisticRest[0][1]


###############################  K nearest neightours   ##############################################


print(parse_one_hot(xin))

np.set_printoptions(precision=2)
importances = classifier.feature_importances_


importances2 = classifier2.feature_importances_
importances2 = ['%.2f'%(float(a)) for a in importances2]

importances3 = classifier3.feature_importances_

s = parse_x(to_print)

pred1 = int(round(y_pred1[0][1]*100))
pred2 = int(round(y_pred2[0][1]*100))
pred3 = int(round(y_pred3[0][1]*100))
pred4 = int(round(y_pred4[0][1]*100))
logistic_pred = int(round(y_pred_logistic[0][1]*100))
#predV2_logistic = int(round(predV2_logistic[0][1]*100))
logistic_mutchups = int(round(logistic_mutchups[0][1]*100))

ensemble_logistic = 0
if games > 70:
    ensemble_logistic = int(round(y_pred_logisticRest[0][1]*100))
else:
    ensemble_logistic = pred3

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ RESULT #############################################################################
print("")
if games < 60:
    logistic_mutchups = 0

log =(s+"-"+str(pred1)
      +"%-"+str(pred2)
      +"%-"+str(pred3)
      +"%-"+str(pred4)
      +"%-"+str(predComboLeanring)
      +"%-"+str(logistic_mutchups)
      +"%-"+str(logistic_pred)
      +"%-"+str(int(round(correct_logistic*100)))
      +"%-"+str(rf_trasformed)
      +"%-"+str(0)
      +"%-\n" )

print(log)


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ RESULT #############################################################################
print("combo importances")
print(combo_importances)
avg_prediction = (y_pred1[0][1]*100 + y_pred2[0][1]*100 + y_pred3[0][1]*100 + y_pred4[0][1]*100)/4
print("strong logistic: " + str(int(round(correct_logistic*100)))+"%")
print("random forests t-winrates: "+ str(rf_trasformed)+"%")
print("random forests winrates: " + str(predComboLeanring) + "%")
print("matchups logistic: " + str(logistic_mutchups)+"%")
print("normal logistic: " + str(logistic_pred)+"%")

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