from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
import numpy as np
import config

from neural_nets.input_to_onehot import input_to_onehot
from neural_nets.models.cross_entropy import input_cross_entropy
from neural_nets.models.cross_entropy.input_cross_entropy import cross_entropy_input_to_onehot, get_predictions

def cluster_bayesian_gmm(onehot_input):
    bgmm = BayesianGaussianMixture(n_components=10).fit(onehot_input)
    input = bgmm.predict_proba(onehot_input)
    input = np.array(input)

    return input


def cluster_gmm(onehot_input):
    gmm = GaussianMixture(n_components=100).fit(onehot_input)
    input = gmm.predict_proba(onehot_input)
    input = np.array(input)

    return input

def parse_x(xin, result):

    data = str(result)+"-"
    xin = [int(x) for x in xin]

    data += config.races[xin[0]]+"-"
    data += config.tryhard[xin[1]]
    data += config.races[xin[2]]+"-"
    data += str(xin[3])
    data += "-"+str(xin[4])+"-"

    if(len(xin)>5):
        data += config.maps[xin[5]]

    return data


onehot_input, y, non_standardized_input = cross_entropy_input_to_onehot()
data = get_predictions()
clustered_input = cluster_bayesian_gmm(non_standardized_input)

wins = np.zeros(len(clustered_input[0]))
games = np.zeros(len(clustered_input[0]))

for i in range(len(clustered_input)):
    print(data[i])
    for cluster in range(len(clustered_input[i])):
        if clustered_input[i][cluster] > 0.5:
            if data[i][0] == 1:
                wins[cluster] += 1

            games[cluster] += 1


for cluster in range(len(clustered_input[0])):
    print("")
    for i in range(len(clustered_input)):
        if clustered_input[i][cluster] > 0.5:
            print(data[i])


print("wins")
winrates = [wins[i]/games[i] for i in range(len(wins))]
print(winrates)
print("")
print("games")
print(games)


