
race_dict = {0: 'Hum', 1: 'Ne', 2: 'Orc', 3: 'Ra', 4: 'Ud'}
map_dict = {0: 'amazonia', 1: 'concealed', 2: 'echo', 3: 'northren', 4: 'refuge', 5: 'swamped', 6: 'terenas', 7: 'turtle', 8: 'twisted'}

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



import statistics
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

################################## Logistic  -------------------------------------------------

globa_grubby_winrates = {'Hum': 0.82, 'Ne': 0.86, 'Orc': 0.87, 'Ra': 0.8, 'Ud': 0.78}

def get_input(enable_formula = False):
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
        power = 2

        few_games_coeff = 1

        if int(X[5]) < 15:
            few_games_coeff = 0.7
        elif int(X[5]) < 40:
            few_games_coeff = 0.9

        if len(contents) - counter < 20:
            few_games_coeff += 0.1
        elif len(contents) - counter < 50:
            few_games_coeff += 0.06

        win_scenario = ((1 - (grubb_wr - opp_wr)) ** power) * result
        lose_scenario = ((1 - (opp_wr - grubb_wr)) ** power) * (1 - result)
        formula = (win_scenario + lose_scenario) * few_games_coeff

        if not enable_formula:
            formula = 1 * few_games_coeff

        satur = 0.1

        race_games[X[1]] += ((1 * formula) )
        race_wins[X[1]] += (int(X[0]) * formula)

        opponent_race_games[X[3]] += ((1 * formula))
        opponent_race_wins[X[3]] += (int(X[0]) * formula)

        map_games[X[6]] += ((1 * formula))
        map_wins[X[6]] += (int(X[0]) * formula)

        grubb_race = X[1]
        opp_race = X[3]
        map = X[6]

        matchup_games[grubb_race][opp_race][map] += ((1 * formula) + satur)
        matchup_wins[grubb_race][opp_race][map] += (int(X[0]) * formula)

        print("g: " + grubb_race + " o: " + opp_race + " map: " + map + " opp wr games: " + X[4]+"-"+X[5]+ " f: "+ str(formula)+" win: "+str(win_scenario)+" lose: "+str(lose_scenario)+" coeff: "+str(few_games_coeff) +" res: "+X[0])

        counter += 1
        X = np.array(X)
        data.append(X)

    return data

def fill_winrates_dictionary():

    for key in race_wins.keys():
        race_winrates[key] = (round((race_wins[key] / race_games[key]) * 10000)) / 10000
        opponent_race_winrates[key] = (round((opponent_race_wins[key] / opponent_race_games[key]) * 10000)) / 10000

    for key in map_wins.keys():
        maps_winrates[key] = (round((map_wins[key] / map_games[key]) * 10000)) / 10000

    saturation = 4
    epsilon = 5
    for grubb_race in matchup_wins.keys():
        for opp_race in matchup_wins[grubb_race].keys():
            for map in matchup_wins[grubb_race][opp_race].keys():
                a = statistics.mean(matchup_winrates[grubb_race][opp_race].values())
                b = maps_winrates[map]
                if matchup_games[grubb_race][opp_race][map] == 0:
                    matchup_winrates[grubb_race][opp_race][map] = 0.8 * a + 0.2 * b
                else:
                    matchup_winrates[grubb_race][opp_race][map] = (round(((matchup_wins[grubb_race][opp_race][map]) / (matchup_games[grubb_race][opp_race][map])) * 10000)) / 10000
                    print(grubb_race + " " + opp_race + " " + map + " winrate " + str((round(((matchup_wins[grubb_race][
                        opp_race][map]) / (matchup_games[grubb_race][opp_race][map])) * 10000)) / 10000))


    print("RACE winrates: " + str(race_winrates))
    print("oppenent race winrates: " + str(opponent_race_winrates))
    print("maps winrates: " + str(maps_winrates))

def transform_input(input):
    labelencoder = LabelEncoder()

    transformed_input = []
    for i in range(len(input)):
        transformed_instance = []
        input[:, 1] = labelencoder.fit_transform(input[:, 1])

        grubb_race = input[i][0]
        opp_race = input[i][2]
        map = input[i][5]

        transformed_instance.append(race_winrates[input[i][0]])
        transformed_instance.append(opponent_race_winrates[input[i][2]])
        transformed_instance.append(maps_winrates[input[i][5]])
        transformed_instance.append(statistics.mean(matchup_winrates[grubb_race][opp_race].values()))
        #transformed_instance.append(matchup_winrates[grubb_race][opp_race][map])
        transformed_instance.append(input[i][1])
        transformed_instance.append(input[i][3])

        transformed_input.append(transformed_instance)

    return transformed_input


def logistic_reg(xin, formula):
    races = {0: 'Hum', 1: 'Ne', 2: 'Orc', 3: 'Ra', 4: 'Ud'}

    maps = {0: 'amazonia', 1: 'concealed', 2: 'echo', 3: 'northren', 4: 'refuge', 5: 'swamped', 6: 'terenas',
            7: 'turtle', 8: 'twisted'}

    input = get_input(formula)
    fill_winrates_dictionary()

    input = np.array(input)

    y = input[:, 0]
    input = input[:, 1:]
    input = transform_input(input)

    clf = LogisticRegression(solver='lbfgs', max_iter=500).fit(input, y)

    Grubby_race = race_dict[xin[0]]
    opponent_race = race_dict[xin[2]]
    map = map_dict[xin[5]]

    t = [races[xin[0]], xin[1], races[xin[2]], xin[3], xin[4], maps[xin[5]]]

    transformed_xin = []
    transformed_xin.append(race_winrates[t[0]])
    transformed_xin.append(opponent_race_winrates[t[2]])
    transformed_xin.append(maps_winrates[t[5]])
    transformed_xin.append(statistics.mean(matchup_winrates[Grubby_race][opponent_race].values()))
    #transformed_xin.append(matchup_winrates[Grubby_race][opponent_race][map])
    transformed_xin.append(t[1])
    transformed_xin.append(t[3])

    print("transformed xin " + str(transformed_xin))

    print("Logistic regression preprocessed")
    y_pred_logistic, _ = clf.predict_proba([transformed_xin])
    print(y_pred_logistic)

    return y_pred_logistic


xin = [1, 1, 1, 67, 540, 0]
pred = logistic_reg(xin, False)