import numpy as np

f = open("Grubb.txt", "r")
contents = f.readlines()

input = []
counter = 0
x_train = []

file = open("GrubbC.txt", "a")



for l in contents:

    x = l.rstrip("\n")
    x = x.replace('-', ',')
    file.write(x + "\n")
    x_train.append(x)

print(x_train)
file.close()