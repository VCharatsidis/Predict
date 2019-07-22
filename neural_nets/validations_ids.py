import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import copy
import os

maps = []


def get_validation_ids():
    script_directory = os.path.split(os.path.abspath(__file__))[0]
    filepath = '..\\logs\\'
    targets = os.path.join(script_directory, filepath + 'automagic.txt')
    f = open(targets, "r")

    val_ids = []
    contents = f.readlines()
    counter = 0

    for line in contents:
        X = line.split('-')

        if len(X) > 18:
            vag = copy.deepcopy(X[16])
            vag = vag.rstrip("\n")
            vag = vag.replace('%', '')
            vag = int(vag)
            if vag != 0:
                val_ids.append(counter)

        counter += 1

    return val_ids


def center(X):
    newX = X - np.mean(X, axis=0)
    return newX


def standardize(X):
    newX = center(X) / np.std(X, axis=0)
    return newX


def filter(list):
    filter_55 = []
    for i in range(len(list)):
        z = list[i].split("-")
        if int(z[4]) < 55:
            continue
        filter_55.append(list[i])

    return filter_55


def print_validation():
    automagic = open("../logs/automagic.txt", "r")
    grubb = open("../logs/Grubb.txt", "r")

    contents_automagic = automagic.readlines()
    contents_grubb = grubb.readlines()

    val_ids = get_validation_ids()
    for i in val_ids:
        print(contents_automagic[i])
        print(contents_grubb[i])

    print(len(val_ids[-150:]))
    print(val_ids[-150:])

def check_input():
    automagic = open("../logs/automagic.txt", "r")
    grubb = open("../logs/Grubb.txt", "r")

    contents_automagic = automagic.readlines()
    contents_grubb = grubb.readlines()

    filtered_automagic = filter(contents_automagic)
    filtered_grubb = filter(contents_grubb)

    for i in range(len(filtered_automagic)):
        print(i)
        print(filtered_automagic[i])
        print(filtered_grubb[i])

        split_automagic = filtered_automagic[i].split("-")
        split_grubb = filtered_grubb[i].split("-")

        for j in range(len(split_grubb)-1):
            if split_automagic[j] != split_grubb[j]:
                print(split_grubb[j])
                print(split_automagic[j])
                print("ERROR")
                print(i)

                break


# check_input()
# print_validation()