import numpy as np
import json
import pygame
import OpenGL
# from pygame.locals import *
# from OpenGL.GL import *
# from OpenGL.GLU import *
# import pywavefront

with open('E:\\Python\\Задание1\\parameters.json', 'r') as f:
    param = json.load(f)

C = []
for c_i in param['c']:
    C.append(c_i['c_i'])
print(C)

Lamb = []
for lamb_i in param['lamb']:
    Lamb.append(lamb_i['lamb_i'])
print(Lamb)

E = []
for e_i in param['e']:
    E.append(e_i['e_i'])
print(E)

Q = []
for Q_i in param['Q']:
    Q.append(Q_i['Q_i'])
print(Q)

count = len(C)
Part_i = np.zeros(count)

Parts = []
for i in range(count):
    Parts.append([])
    for vf in range(2):
        Parts[i].append([])

for line in open('E:\\Python\\Задание1\\model1.obj', "r"):
    values = line.split()
    if not values: continue

    if len(values) == 3:
        if values[2] == 'Part1':
            for i in range(count): Part_i[i] = 0
            Part_i[0] = 1
        elif values[2] == 'Part2':
            for i in range(count): Part_i[i] = 0
            Part_i[1] = 1
        elif values[2] == 'Part3':
            for i in range(count): Part_i[i] = 0
            Part_i[2] = 1
        elif values[2] == 'Part4':
            for i in range(count): Part_i[i] = 0
            Part_i[3] = 1
        elif values[2] == 'Part5':
            for i in range(count): Part_i[i] = 0
            Part_i[4] = 1
        else: continue

    if values[0] == 'v':
        index = 0
        for i in range(count):
            if Part_i[i] == 1:
                index = i
        Parts[index][0].append((values[1:4]))

    if values[0] == 'f':
        index = 0
        for i in range(count):
            if Part_i[i] == 1:
                index = i
        Parts[index][1].append((values[1:4]))


print(Parts[1][1])