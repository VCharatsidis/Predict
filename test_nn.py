import torch
import numpy as np
from torch.autograd import Variable
from input_to_onehot import input_to_onehot


def one_hot(xin):
    onehot_encoded = []

    number_races = 5
    tryhard_or_not = 2
    maps = 10

    one_h(onehot_encoded, number_races, xin[0])
    one_h(onehot_encoded, tryhard_or_not, xin[1])
    one_h(onehot_encoded, number_races, xin[2])
    one_h(onehot_encoded, maps, xin[5])

    onehot_encoded = np.array(onehot_encoded)
    onehot_encoded.flatten()
    onehot_encoded = np.append(onehot_encoded, xin[3])
    onehot_encoded = np.append(onehot_encoded, xin[4])

    onehot_encoded.flatten()

    return onehot_encoded


def get_means_and_stds():
    _, _, X_train = input_to_onehot()

    mean1 = np.mean(X_train[:, -1], axis=0)
    std1 = np.std(X_train[:, -1], axis=0)

    mean2 = np.mean(X_train[:, -2], axis=0)
    std2 = np.std(X_train[:, -2], axis=0)

    return mean1, std1, mean2, std2


def standardize(onehot_neural, mean1, std1, mean2, std2):
    onehot_neural = onehot_neural.astype(float)
    onehot_neural[-1] = min(onehot_neural[-1], 200.)

    onehot_neural[-1] -= mean1
    k = onehot_neural[-1] / std1
    onehot_neural[-1] = k

    onehot_neural[-2] -= mean2
    z = onehot_neural[-2] / std2
    onehot_neural[-2] = z

    xin = Variable(torch.FloatTensor([onehot_neural]))

    return xin


def one_h(onehot_encoded, categories, xin):
    letter = [0 for _ in range(categories)]
    letter[xin] = 1
    for i in letter:
        onehot_encoded.append(i)


def test(modelTest, x):
    predTest = modelTest.forward(x)
    neural_predTest = predTest.detach().numpy()

    return str(str(int(round(neural_predTest[0][0]*100))))


def test_all(model):
    print(model)
    modelTest = torch.load(model)
    modelTest.eval()

    ud = [4, 1, 4, 98, 100, 1]
    ud5 = [4, 1, 4, 95, 20, 1]
    ud4 = [4, 1, 4, 98, 150, 6]
    ud1 = [4, 1, 4, 95, 100, 3]
    ud2 = [4, 1, 4, 95, 3000, 3]
    ud3 = [4, 1, 4, 90, 10, 3]

    orc = [2, 1, 2, 80, 180, 2]
    orc2 = [2, 1, 2, 60, 150, 2]
    orc3 = [2, 1, 4, 98, 1200, 1]
    orc4 = [2, 1, 4, 98, 13, 1]
    orc5 = [2, 1, 1, 90, 1900, 6]

    hum = [0, 1, 0, 88, 3500, 9]
    hum1 = [0, 1, 0, 85, 3500, 4]
    hum3 = [0, 0, 4, 72, 165, 0]

    ne = [1, 1, 4, 100, 6, 6]

    undeads = []
    undeads.append(ud)
    undeads.append(ud1)
    undeads.append(ud2)
    undeads.append(ud3)
    undeads.append(ud4)
    undeads.append(ud5)

    orcs = []
    orcs.append(orc)
    orcs.append(orc2)
    orcs.append(orc3)
    orcs.append(orc4)
    orcs.append(orc5)

    humans = []
    humans.append(hum)
    humans.append(hum1)
    humans.append(hum3)

    nes = []
    nes.append(ne)

    mean1, std1, mean2, std2 = get_means_and_stds()

    print_results(undeads, mean1, std1, mean2, std2, modelTest)
    print(" ")
    print_results(orcs, mean1, std1, mean2, std2, modelTest)
    print(" ")
    print_results(humans, mean1, std1, mean2, std2, modelTest)
    print(" ")
    print_results(nes, mean1, std1, mean2, std2, modelTest)


def print_results(undeads, mean1, std1, mean2, std2, model):
    for x in undeads:
        one_x = one_hot(x)
        standardize_x = standardize(one_x, mean1, std1, mean2, std2)

        pred = test(model, standardize_x)
        parse_x(x, pred)


def parse_x(xin, prediction):
    races = {0: 'Hum-', 1: 'Ne-', 2: 'Orc-', 3: 'Ra-', 4: 'Ud-'}

    maps = {0: 'amazonia', 1: 'concealed', 2: 'echo', 3: 'northren', 4: 'refuge', 5: 'swamped', 6: 'terenas',
            7: 'turtle', 8: 'twisted', 9: 'ancient'}

    tryhard = {0: 'r-', 1: 't-'}

    data = ""
    data += races[xin[0]]
    data += tryhard[xin[1]]
    data += races[xin[2]]
    data += str(xin[3])
    data += "-"+str(xin[4])+"-"
    data += maps[xin[5]]+"-"
    data += prediction+"%"

    print(data)
    return data


def test_all_models():
    models = ['grubbyStar.model', 'grubbyStar2.model', 'grubbyStar3L-3W.model', 'grubbyStar4L-3W.model']
    for model in models:
        test_all(model)


test_all_models()