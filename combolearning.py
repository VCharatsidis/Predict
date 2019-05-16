import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import copy
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


f = open("Grubb.txt", "r")
contents = f.readlines()
input = []

race_games = {'Hum': 0, 'Ne': 0, 'Orc': 0, 'Ra': 0, 'Ud': 0}
race_wins = {'Hum': 0, 'Ne': 0, 'Orc': 0, 'Ra': 0, 'Ud': 0}
race_winrates = {'Hum': 0, 'Ne': 0, 'Orc': 0, 'Ra': 0, 'Ud': 0}

opponent_race_games = {'Hum': 0, 'Ne': 0, 'Orc': 0, 'Ra': 0, 'Ud': 0}
opponent_race_wins = {'Hum': 0, 'Ne': 0, 'Orc': 0, 'Ra': 0, 'Ud': 0}
opponent_race_winrates = {'Hum': 0, 'Ne': 0, 'Orc': 0, 'Ra': 0, 'Ud': 0}

map_games = {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0,
             'turtle': 0, 'twisted': 0}

map_wins = {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0,
             'turtle': 0, 'twisted': 0}

maps_winrates = {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0,
                'turtle': 0, 'twisted': 0}

def get_input():
    for l in contents:
        X = l.split('-')

        X[4] = int(X[4])
        X[5] = int(X[5])
        X[6] = X[6].rstrip("\n")

        race_games[X[1]] += 1
        race_wins[X[1]] += int(X[0])

        opponent_race_games[X[3]] += 1
        opponent_race_wins[X[3]] += int(X[0])

        map_games[X[6]] += 1
        map_wins[X[6]] += int(X[0])

        X = np.array(X)
        input.append(X)

    return input


def fill_dictionaries():
    for key in race_wins.keys():
        race_winrates[key] = (round((race_wins[key] / race_games[key]) * 10000)) / 10000
        opponent_race_winrates[key] = (round((opponent_race_wins[key] / opponent_race_games[key]) * 10000)) / 10000

    print("race winrates: " + str(race_winrates))
    print("oppenent race winrates: " + str(opponent_race_winrates))

    for key in map_wins.keys():
        maps_winrates[key] = (round((map_wins[key] / map_games[key]) * 10000)) / 10000

    print("maps winrates: " + str(maps_winrates))
    print("maps games: " + str(map_games))


def xin_to_onehot(xin):
    letter = [0 for _ in range(5)]
    letter[xin[0]] = 1

    xin_onehot = []
    for i in letter:
        xin_onehot.append(i)

    xin_onehot = np.array(xin_onehot)
    xin_onehot.flatten()

    return xin_onehot


def input_to_onehot(input):
    input = np.array(input)
    y = input[:, 0]
    labelencoder = LabelEncoder()

    input = input[:, 1]
    input = labelencoder.fit_transform(input)
    input = [int(x) for x in input]
    print(input)

    onehot_input = np.zeros((len(input), 5))
    onehot_input[np.arange(len(input)), input] = 1

    print(onehot_input)
    return onehot_input, y




def transform_input(input):
    labelencoder = LabelEncoder()

    for i in range(len(input)):
        input[:, 1] = labelencoder.fit_transform(input[:, 1])

        input[i] = [race_winrates[input[i][0]], input[i][1],
                    opponent_race_winrates[input[i][2]], int(input[i][3]) / 100,
                    int(input[i][4]), maps_winrates[input[i][5]]]

    return input

def predict(xin):
    races = {0: 'Hum', 1: 'Ne', 2: 'Orc', 3: 'Ra', 4: 'Ud'}

    maps = {0: 'amazonia', 1: 'concealed', 2: 'echo', 3: 'northren', 4: 'refuge', 5: 'swamped', 6: 'terenas',
            7: 'turtle', 8: 'twisted'}

    input = get_input()

    fill_dictionaries()
    t = [races[xin[0]], xin[1], races[xin[2]], xin[3], xin[4], maps[xin[5]]]
    xin_processed = [race_winrates[t[0]], t[1], opponent_race_winrates[t[2]], t[3], t[4], maps_winrates[t[5]]]
    input = np.array(input)
    y = input[:, 0]

    input = input[:, 1:]
    input = transform_input(input)


    estimators = 500
    classifier = RandomForestClassifier(n_estimators=estimators, random_state=0, oob_score=True)
    classifier.fit(input, y)
    importances = classifier.feature_importances_

    prediction = classifier.predict_proba([xin_processed])
    prediction = int(round(prediction[0][1]*100))

    oob_error3 = 1 - classifier.oob_score_
    print(oob_error3)

    return prediction, importances


prediction = predict([0, 1, 1, 90, 1800, 1])
print(str(prediction) + "%")