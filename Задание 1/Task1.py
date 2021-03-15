import numpy as np
import math
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
        Parts[index][0].append(list(map(float, values[1:4])))

    if values[0] == 'f':
        index = 0
        for i in range(count):
            if Part_i[i] == 1:
                index = i
    
        if len(Parts[index][1]) == 0:
            err = int(values[1]) - 1
        for i in range(len(values) - 1):
            values[i + 1] = int(values[i + 1]) - err
        Parts[index][1].append(list(map(int, values[1:4])))

# print(Parts[4][1])

S_ij = np.zeros((count, count))
for p in range(len(Parts)):
    for k in range(len(Parts)):
        Index_v_p = np.zeros(len(Parts[p][0]))
        Index_v_k = np.zeros(len(Parts[k][0]))

        Index_f_p = np.zeros(len(Parts[p][1]))
        Index_f_k = np.zeros(len(Parts[k][1]))

        for i in range(len(Parts[p][0])):
            for j in range(len(Parts[k][0])):
                if Parts[p][0][i] == Parts[k][0][j]:
                    Index_v_p[i] += 1
                    Index_v_k[j] += 1

        high = 0.0
        for i in range(len(Index_v_p)):
            if Index_v_p[i] > 0:
                high = Parts[p][0][i][1]
                break

        for i in range(len(Parts[p][0])):
            if Parts[p][0][i][1] == high and Index_v_p[i] == 0:
                Index_v_p[i] += 1
        for i in range(len(Parts[k][0])):
            if Parts[k][0][i][1] == high and Index_v_k[i] == 0:
                Index_v_k[i] += 1                

        for i in range(len(Parts[p][1])):
            for j in range(3):
                if Index_v_p[Parts[p][1][i][j] - 1] == 1:
                    Index_f_p[i] += 1
            if Index_f_p[i] == 3:
                Index_f_p[i] = 1
            else:
                Index_f_p[i] = 0
        for i in range(len(Parts[k][1])):
            for j in range(3):
                if Index_v_k[Parts[k][1][i][j] - 1] == 1:
                    Index_f_k[i] += 1
            if Index_f_k[i] == 3:
                Index_f_k[i] = 1
            else:
                Index_f_k[i] = 0

        S_p = 0
        S_k = 0
        AB = np.zeros(3)
        AC = np.zeros(3)
        for i in range(len(Index_f_p)):
            if Index_f_p[i] == 1:
                AB[0] = Parts[p][0][Parts[p][1][i][1] - 1][0] - Parts[p][0][Parts[p][1][i][0] - 1][0]
                AB[1] = Parts[p][0][Parts[p][1][i][1] - 1][1] - Parts[p][0][Parts[p][1][i][0] - 1][1]
                AB[2] = Parts[p][0][Parts[p][1][i][1] - 1][2] - Parts[p][0][Parts[p][1][i][0] - 1][2]

                AC[0] = Parts[p][0][Parts[p][1][i][2] - 1][0] - Parts[p][0][Parts[p][1][i][0] - 1][0]
                AC[1] = Parts[p][0][Parts[p][1][i][2] - 1][1] - Parts[p][0][Parts[p][1][i][0] - 1][1]
                AC[2] = Parts[p][0][Parts[p][1][i][2] - 1][2] - Parts[p][0][Parts[p][1][i][0] - 1][2]

                ABxAC = np.cross(AB, AC)
                S_p += math.sqrt(ABxAC[0]**2 + ABxAC[1]**2 + ABxAC[2]**2) / 2.0

        for i in range(len(Index_f_k)):
            if Index_f_k[i] >= 1:
                AB[0] = Parts[k][0][Parts[k][1][i][1] - 1][0] - Parts[k][0][Parts[k][1][i][0] - 1][0]
                AB[1] = Parts[k][0][Parts[k][1][i][1] - 1][1] - Parts[k][0][Parts[k][1][i][0] - 1][1]
                AB[2] = Parts[k][0][Parts[k][1][i][1] - 1][2] - Parts[k][0][Parts[k][1][i][0] - 1][2]

                AC[0] = Parts[k][0][Parts[k][1][i][2] - 1][0] - Parts[k][0][Parts[k][1][i][0] - 1][0]
                AC[1] = Parts[k][0][Parts[k][1][i][2] - 1][1] - Parts[k][0][Parts[k][1][i][0] - 1][1]
                AC[2] = Parts[k][0][Parts[k][1][i][2] - 1][2] - Parts[k][0][Parts[k][1][i][0] - 1][2]

                ABxAC = np.cross(AB, AC)
                S_k += math.sqrt(ABxAC[0]**2 + ABxAC[1]**2 + ABxAC[2]**2) / 2

        if S_p > S_k:
            S_ij[p][k] = S_k
        else:
            S_ij[p][k] = S_p

print(S_ij)
            