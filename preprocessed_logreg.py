
race_dict = {0: 'Hum', 1: 'Ne', 2: 'Orc', 3: 'Ra', 4: 'Ud'}
map_dict = {0: 'amazonia', 1: 'concealed', 2: 'echo', 3: 'northren', 4: 'refuge', 5: 'swamped', 6: 'terenas', 7: 'turtle', 8: 'twisted'}

matchup_games = {'Hum': {'Hum': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ne': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Orc': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ra': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ud': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0}},

                  'Ne': {'Hum': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ne': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Orc': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ra': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ud': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0}},

                  'Orc': {'Hum': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ne': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Orc': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ra': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ud': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0}},

                  'Ra': {'Hum': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ne': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Orc': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ra': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ud': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0}},

                  'Ud': {'Hum': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ne': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Orc': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ra': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ud': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0}}}


matchup_wins = {'Hum': {'Hum': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ne': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Orc': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ra': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ud': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0}},

                  'Ne': {'Hum': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ne': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Orc': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ra': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ud': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0}},

                  'Orc': {'Hum': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ne': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Orc': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ra': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ud': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0}},

                  'Ra': {'Hum': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ne': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Orc': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ra': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ud': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0}},

                  'Ud': {'Hum': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ne': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Orc': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ra': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ud': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0}}}

matchup_winrates = {'Hum': {'Hum': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ne': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Orc': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ra': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ud': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0}},

                  'Ne': {'Hum': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ne': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Orc': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ra': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ud': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0}},

                  'Orc': {'Hum': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ne': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Orc': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ra': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ud': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0}},

                  'Ra': {'Hum': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ne': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Orc': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ra': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ud': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0}},

                  'Ud': {'Hum': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ne': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Orc': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ra': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0},
                         'Ud': {'amazonia': 0, 'concealed': 0, 'echo': 0, 'northren': 0, 'refuge': 0, 'swamped': 0, 'terenas': 0, 'turtle': 0, 'twisted': 0}}}




import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

################################## Logistic  -------------------------------------------------

globa_grubby_winrates = {'Hum': 0.82, 'Ne': 0.86, 'Orc': 0.87, 'Ra': 0.8, 'Ud': 0.78}

def get_input():
    f = open("Grubb.txt", "r")

    contents = f.readlines()

    data = []
    counter = 0
    for l in contents:
        X = l.split('-')

        X[6] = X[6].rstrip("\n")

        grubb_wr = globa_grubby_winrates[X[1]]
        opp_wr = int(X[4]) / 100
        result = int(X[0])
        power = 4

        coeff = 1

        if int(X[5]) < 15:
            coeff = 0.3
        elif int(X[5]) < 40:
            coeff = 0.5

        if len(contents) - counter < 15:
            coeff += 0.1

        win_scenario = ((1 - (grubb_wr - opp_wr)) ** power) * result
        lose_scenario = ((1 - (opp_wr - grubb_wr)) ** power) * (1 - result)
        formula = (win_scenario + lose_scenario) * coeff

        grubb_race = X[1]
        opp_race = X[3]
        map = X[6]

        matchup_games[grubb_race][opp_race][map] += (1 * formula)
        matchup_wins[grubb_race][opp_race][map] += (int(X[0]) * formula)

        print("g: " + grubb_race + " o: " + opp_race + " map: " + map + " opp wr games: " + X[4]+"-"+X[5]+ " f: "+ str(formula)+" win: "+str(win_scenario)+" lose: "+str(lose_scenario)+" coeff: "+str(coeff) +" res: "+X[0])

        counter += 1
        X = np.array(X)
        data.append(X)

    return data

def fill_winrates_dictionary():
    epsilon = 0.1
    for grubb_race in matchup_wins.keys():
        for opp_race  in matchup_wins[grubb_race].keys():
            for map in matchup_wins[grubb_race][opp_race].keys():
                print(matchup_wins[grubb_race][opp_race][map])
                print(matchup_games[grubb_race][opp_race][map])
                matchup_winrates[grubb_race][opp_race][map] = (round(((matchup_wins[grubb_race][opp_race][map]+0.05) / (matchup_games[grubb_race][opp_race][map] + epsilon)) * 10000)) / 10000
                print(grubb_race + " " + opp_race + " " + map + " winrate " + str((round(((matchup_wins[grubb_race][opp_race][map] + 0.05) / (matchup_games[grubb_race][opp_race][map] + epsilon)) * 10000)) / 10000))


def transform_input(input):
    labelencoder = LabelEncoder()

    transformed_input = []
    for i in range(len(input)):
        transformed_instance = []
        input[:, 1] = labelencoder.fit_transform(input[:, 1])

        grubb_race = input[i][0]
        opp_race = input[i][2]
        map = input[i][5]

        transformed_instance.append(matchup_winrates[grubb_race][opp_race][map])
        transformed_instance.append(input[i][1])
        transformed_instance.append(input[i][3])

        print(transformed_instance)
        transformed_input.append(transformed_instance)

    return transformed_input


def logistic_reg(xin):

    input = get_input()
    fill_winrates_dictionary()

    input = np.array(input)

    print(input)
    y = input[:, 0]
    input = input[:, 1:]
    input = transform_input(input)
    print(input)

    print("Logistic regression train")
    clf = LogisticRegression(solver='lbfgs', max_iter=500).fit(input, y)

    Grubby_race = race_dict[xin[0]]
    opponent_race = race_dict[xin[2]]
    map = map_dict[xin[5]]
    xin = [matchup_winrates[Grubby_race][opponent_race][map], xin[1], xin[3]]


    print("xin " + str(xin))

    print("Logistic regression preprocessed")
    y_pred_logistic,_ = clf.predict_proba([xin])
    print(y_pred_logistic)

    return y_pred_logistic


xin = [2, 1, 4, 98, 1200, 0]
logistic_reg(xin)