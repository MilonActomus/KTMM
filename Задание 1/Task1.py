import numpy as np
import math
import json
import csv
import OpenGL
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtOpenGL

import OpenGL.GL as gl
from OpenGL import GLU

from scipy.integrate import odeint

from scipy.optimize import fsolve
import matplotlib.pyplot as plt

import sys
import time
import ctypes

from PyQt5.QtCore import QThread

class TimeThread(QThread):
    def __init__(self, mainwindow, parent=None):
        super().__init__()
        self.mainwindow = mainwindow

    def run(self):
        f = open(self.mainwindow.Tname.text(), "w+")
        f.close()

        self.mainwindow.glWidget.T0 = []

        for t_i in range(int(self.mainwindow.Time_param.text())):
            self.mainwindow.glWidget.t = np.linspace(t_i, t_i+1, 2)
            
            self.mainwindow.glWidget.Parsing(self.mainwindow.Mname.text(), self.mainwindow.Pname.text(), self.mainwindow.Tname.text(),\
                                                                                int(self.mainwindow.Tmax.text()), int(self.mainwindow.Tmin.text()))
            
            if t_i == 0:
                self.mainwindow.Time_.setText(str(t_i))
                self.mainwindow.slider.setValue(int(self.mainwindow.Time_.text()))
            
            self.mainwindow.Time_.setText(str(t_i + 1))
            self.mainwindow.slider.setValue(int(self.mainwindow.Time_.text()))
            time.sleep(0.1)
            
           

class GLWidget(QtOpenGL.QGLWidget):
    def __init__(self, parent=None):
        self.parent = parent
        QtOpenGL.QGLWidget.__init__(self, parent)
            
    def initializeGL(self):
        self.qglClearColor(QtGui.QColor(255, 255, 255))
        gl.glEnable(gl.GL_DEPTH_TEST)

        self.Parts = []
        self.T = []
        self.T0 = []
        self.t = []

        f = open(self.parent.Tname.text(), "w+")
        f.close()

        self.b = 20
        self.c = 3
        self.d = 4

        self.qe = 100
        self.deg = 4

        self.rotX = 0.0
        self.rotY = 0.0
        self.rotZ = 0.0
         
    def resizeGL(self, width, height):
        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        aspect = width / float(height)

        GLU.gluPerspective(45.0, aspect, 1.0, 100.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)

    def paintGL(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        
        for i in range(16):
            gl.glBegin(gl.GL_QUADS)
            gl.glColor3f (1.0 - i*16/256, 0.0, 0.0 + i*16/256)
            gl.glVertex3fv([-5.25, 4.0 - i*0.5, -10.0])
            gl.glVertex3fv([-4.75, 4.0 - i*0.5, -10.0])
            gl.glVertex3fv([-4.75, 3.5-  i*0.5, -10.0])
            gl.glVertex3fv([-5.25, 3.5 - i*0.5, -10.0])
            gl.glEnd()

        gl.glPushMatrix()

        gl.glTranslate(0.0, -5.0, -40.0)
        # gl.glScale(20.0, 20.0, 20.0)
        gl.glRotate(self.rotX, 1.0, 0.0, 0.0)
        gl.glRotate(self.rotY, 0.0, 1.0, 0.0)
        gl.glRotate(self.rotZ, 0.0, 0.0, 1.0)
        # gl.glTranslate(-0.5, -0.5, -0.5)

        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glEnableClientState(gl.GL_COLOR_ARRAY)
        
        if len(self.Parts) != 0:
            buffer_offset = ctypes.c_void_p
            float_size = ctypes.sizeof(ctypes.c_float)

            gl.glVertexPointer(3, gl.GL_FLOAT, 24, buffer_offset(self.VtxArray.ctypes.data))
            gl.glColorPointer(3, gl.GL_FLOAT, 24, buffer_offset(self.VtxArray.ctypes.data + float_size * 3))
            
            for i in range(self.count):
                gl.glDrawElements(gl.GL_TRIANGLES, len(self.IdxArray[i]), gl.GL_UNSIGNED_INT, self.IdxArray[i])

        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        gl.glDisableClientState(gl.GL_COLOR_ARRAY)

        gl.glPopMatrix()

    def Parsing(self, Mname, Pname, Tname, Tmax, Tmin):
        if len(self.Parts) == 0:
            self.count = 5
            Part_i = np.zeros(self.count)

            self.Parts = []
            for i in range(self.count):
                self.Parts.append([])
                for vf in range(2):
                    self.Parts[i].append([])
            
            for line in open(Mname, "r"):
                values = line.split()
                if not values: continue

                if len(values) == 3:
                    if values[2] == 'Part1':
                        for i in range(self.count): Part_i[i] = 0
                        Part_i[0] = 1
                    elif values[2] == 'Part2':
                        for i in range(self.count): Part_i[i] = 0
                        Part_i[1] = 1
                    elif values[2] == 'Part3':
                        for i in range(self.count): Part_i[i] = 0
                        Part_i[2] = 1
                    elif values[2] == 'Part4':
                        for i in range(self.count): Part_i[i] = 0
                        Part_i[3] = 1
                    elif values[2] == 'Part5':
                        for i in range(self.count): Part_i[i] = 0
                        Part_i[4] = 1
                    else: continue

                if values[0] == 'v':
                    index = 0
                    for i in range(self.count):
                        if Part_i[i] == 1:
                            index = i
                    self.Parts[index][0].append(list(map(float, values[1:4])))

                if values[0] == 'f':
                    index = 0
                    for i in range(self.count):
                        if Part_i[i] == 1:
                            index = i
                
                    if len(self.Parts[index][1]) == 0:
                        err = 1
                    for i in range(len(values) - 1):
                        values[i + 1] = int(values[i + 1]) - err
                    self.Parts[index][1].append(list(map(int, values[1:4])))
            
            self.S_ij = np.zeros((self.count, self.count))
            for p in range(len(self.Parts)):
                for k in range(len(self.Parts)):
                    Index_v_p = np.zeros(len(self.Parts[p][0]))
                    Index_v_k = np.zeros(len(self.Parts[k][0]))

                    Index_f_p = np.zeros(len(self.Parts[p][1]))
                    Index_f_k = np.zeros(len(self.Parts[k][1]))

                    for i in range(len(self.Parts[p][0])):
                        for j in range(len(self.Parts[k][0])):
                            if self.Parts[p][0][i] == self.Parts[k][0][j]:
                                Index_v_p[i] += 1
                                Index_v_k[j] += 1

                    high = 0.0
                    for i in range(len(Index_v_p)):
                        if Index_v_p[i] > 0:
                            high = self.Parts[p][0][i][1]
                            break

                    for i in range(len(self.Parts[p][0])):
                        if self.Parts[p][0][i][1] == high and Index_v_p[i] == 0:
                            Index_v_p[i] += 1
                    for i in range(len(self.Parts[k][0])):
                        if self.Parts[k][0][i][1] == high and Index_v_k[i] == 0:
                            Index_v_k[i] += 1                

                    for i in range(len(self.Parts[p][1])):
                        for j in range(3):
                            if Index_v_p[self.Parts[p][1][i][j] - self.Parts[p][1][0][0]] == 1:
                                Index_f_p[i] += 1
                        if Index_f_p[i] == 3:
                            Index_f_p[i] = 1
                        else:
                            Index_f_p[i] = 0
                    for i in range(len(self.Parts[k][1])):
                        for j in range(3):
                            if Index_v_k[self.Parts[k][1][i][j] - self.Parts[k][1][0][0]] == 1:
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
                            AB[0] = self.Parts[p][0][self.Parts[p][1][i][1] - self.Parts[p][1][0][0]][0] - self.Parts[p][0][self.Parts[p][1][i][0] - self.Parts[p][1][0][0]][0]
                            AB[1] = self.Parts[p][0][self.Parts[p][1][i][1] - self.Parts[p][1][0][0]][1] - self.Parts[p][0][self.Parts[p][1][i][0] - self.Parts[p][1][0][0]][1]
                            AB[2] = self.Parts[p][0][self.Parts[p][1][i][1] - self.Parts[p][1][0][0]][2] - self.Parts[p][0][self.Parts[p][1][i][0] - self.Parts[p][1][0][0]][2]

                            AC[0] = self.Parts[p][0][self.Parts[p][1][i][2] - self.Parts[p][1][0][0]][0] - self.Parts[p][0][self.Parts[p][1][i][0] - self.Parts[p][1][0][0]][0]
                            AC[1] = self.Parts[p][0][self.Parts[p][1][i][2] - self.Parts[p][1][0][0]][1] - self.Parts[p][0][self.Parts[p][1][i][0] - self.Parts[p][1][0][0]][1]
                            AC[2] = self.Parts[p][0][self.Parts[p][1][i][2] - self.Parts[p][1][0][0]][2] - self.Parts[p][0][self.Parts[p][1][i][0] - self.Parts[p][1][0][0]][2]

                            ABxAC = np.cross(AB, AC)
                            S_p += math.sqrt(ABxAC[0]**2 + ABxAC[1]**2 + ABxAC[2]**2) / 2.0

                    for i in range(len(Index_f_k)):
                        if Index_f_k[i] >= 1:
                            AB[0] = self.Parts[k][0][self.Parts[k][1][i][1] - self.Parts[k][1][0][0]][0] - self.Parts[k][0][self.Parts[k][1][i][0] - self.Parts[k][1][0][0]][0]
                            AB[1] = self.Parts[k][0][self.Parts[k][1][i][1] - self.Parts[k][1][0][0]][1] - self.Parts[k][0][self.Parts[k][1][i][0] - self.Parts[k][1][0][0]][1]
                            AB[2] = self.Parts[k][0][self.Parts[k][1][i][1] - self.Parts[k][1][0][0]][2] - self.Parts[k][0][self.Parts[k][1][i][0] - self.Parts[k][1][0][0]][2]

                            AC[0] = self.Parts[k][0][self.Parts[k][1][i][2] - self.Parts[k][1][0][0]][0] - self.Parts[k][0][self.Parts[k][1][i][0] - self.Parts[k][1][0][0]][0]
                            AC[1] = self.Parts[k][0][self.Parts[k][1][i][2] - self.Parts[k][1][0][0]][1] - self.Parts[k][0][self.Parts[k][1][i][0] - self.Parts[k][1][0][0]][1]
                            AC[2] = self.Parts[k][0][self.Parts[k][1][i][2] - self.Parts[k][1][0][0]][2] - self.Parts[k][0][self.Parts[k][1][i][0] - self.Parts[k][1][0][0]][2]

                            ABxAC = np.cross(AB, AC)
                            S_k += math.sqrt(ABxAC[0]**2 + ABxAC[1]**2 + ABxAC[2]**2) / 2

                    if S_p > S_k:
                        self.S_ij[p][k] = S_k
                    else:
                        self.S_ij[p][k] = S_p
        
        with open(Pname, 'r') as f:
            param = json.load(f)

        self.C = []
        for c_i in param['c']:
            self.C.append(c_i['c_i'])

        self.Lamb = []
        for lamb_i in param['lamb']:
            self.Lamb.append(lamb_i['lamb_i'])

        self.E = []
        for e_i in param['e']:
            self.E.append(e_i['e_i'])

        self.Q = []
        for Q_i in param['Q']:
            self.Q.append(Q_i['Q_i'])

        self.T = self.Solve(Tname)

    def Solve(self, Tname):
        if len(self.T0) == 0:
            self.T0 = fsolve(self.Stationary_solution, np.zeros(self.count), args=(self.t[0]))

            self.T0[1] = 30.0
            self.T = odeint(self.ODE, self.T0, self.t)

            temp_data = []
            temp_data.append([self.t[0], self.T[0][0], self.T[0][1], self.T[0][2], self.T[0][3], self.T[0][4]])
            myFile = open(Tname, 'a', newline="")
            with myFile:
                writer = csv.writer(myFile, delimiter=';')
                writer.writerows(temp_data)
        else:
            self.T = odeint(self.ODE, self.T[1], self.t)

            temp_data = []
            for i in range(len(self.t) - 1):
                temp_data.append([self.t[i], self.T[i][0], self.T[i][1], self.T[i][2], self.T[i][3], self.T[i][4]])
            myFile = open(Tname, 'a', newline="")
            with myFile:
                writer = csv.writer(myFile, delimiter=';')
                writer.writerows(temp_data)

        return self.T

    def Plot_sol(self):
        with open(self.parent.Tname.text()) as File:
            reader = csv.reader(File, delimiter=';', quotechar=',', quoting=csv.QUOTE_MINIMAL)
            T_plot = []
            for row in reader:
                T_plot.append([float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])])
            
            t_plot = np.linspace(0, int(self.parent.Time_param.text()), int(self.parent.Time_param.text()))
            T_plot = np.array(T_plot)

            plt.plot(t_plot, T_plot[:, 0], 'b', label='Part1')
            plt.plot(t_plot, T_plot[:, 1], 'g', label='Part2')
            plt.plot(t_plot, T_plot[:, 2], 'r', label='Part3')
            plt.plot(t_plot, T_plot[:, 3], 'c', label='Part4')
            plt.plot(t_plot, T_plot[:, 4], 'm', label='Part5')
            plt.legend(loc='best')
            plt.xlabel('t')
            plt.grid()
            plt.show()
        

    def Stationary_solution(self, T, t):
        C0 = 5.67
        return [(self.Lamb[0] * self.S_ij[0][1] * (T[1] - T[0]) - self.E[0] * self.S_ij[0][0] * C0 * (T[0]/self.qe)**self.deg) / self.C[0],
                (self.Lamb[1] * self.S_ij[1][2] * (T[2] - T[1]) + self.Lamb[0] * self.S_ij[1][0] * (T[0] - T[1]) - self.E[1] * self.S_ij[1][1] * C0 * (T[1]/self.qe)**self.deg + \
                                                                                                                self.Q[1] * (self.b + self.c * math.cos(t/self.d))) / self.C[1],
                (self.Lamb[2] * self.S_ij[2][3] * (T[3] - T[2]) + self.Lamb[1] * self.S_ij[2][1] * (T[1] - T[2]) - self.E[2] * self.S_ij[2][2] * C0 * (T[2]/self.qe)**self.deg) / self.C[2],
                (self.Lamb[3] * self.S_ij[3][4] * (T[4] - T[3]) + self.Lamb[2] * self.S_ij[3][2] * (T[2] - T[3]) - self.E[3] * self.S_ij[3][3] * C0 * (T[3]/self.qe)**self.deg) / self.C[3],
                (self.Lamb[3] * self.S_ij[4][3] * (T[3] - T[4]) - self.E[4] * self.S_ij[4][4] * C0 * (T[4]/self.qe)**self.deg) / self.C[4]]

    def ODE(self, T, t):
        T1, T2, T3, T4, T5 = T
        C0 = 5.67
        dTdt = [(self.Lamb[0] * self.S_ij[0][1] * (T2 - T1) - self.E[0] * self.S_ij[0][0] * C0 * (T1/self.qe)**self.deg) / self.C[0],
                (self.Lamb[1] * self.S_ij[1][2] * (T3 - T2) + self.Lamb[0] * self.S_ij[1][0] * (T1 - T2) - self.E[1] * self.S_ij[1][1] * C0 * (T2/self.qe)**self.deg + \
                                                                                                        self.Q[1] * (self.b + self.c * math.cos(t/self.d))) / self.C[1],
                (self.Lamb[2] * self.S_ij[2][3] * (T4 - T3) + self.Lamb[1] * self.S_ij[2][1] * (T2 - T3) - self.E[2] * self.S_ij[2][2] * C0 * (T3/self.qe)**self.deg) / self.C[2],
                (self.Lamb[3] * self.S_ij[3][4] * (T5 - T4) + self.Lamb[2] * self.S_ij[3][2] * (T3 - T4) - self.E[3] * self.S_ij[3][3] * C0 * (T4/self.qe)**self.deg) / self.C[3],
                (self.Lamb[3] * self.S_ij[4][3] * (T4 - T5) - self.E[4] * self.S_ij[4][4] * C0 * (T5/self.qe)**self.deg) / self.C[4]]
        return dTdt

    def initGeometry(self, Tmax, Tmin):
        if Tmax < Tmin:
            c = Tmax
            Tmax = Tmin
            Tmin = c
        
        if len(self.Parts) != 0:
            if self.t[0] == 0:
                for t_i in range(2):
                    Vtx = []
                    for i in range(self.count):
                        for j in range(len(self.Parts[i][0])):
                            Vtx += self.Parts[i][0][j]
                            Vtx += self.red(self.T[t_i][i] - Tmin, Tmax - Tmin)
                    self.VtxArray = np.array(Vtx, dtype = np.float32)

                    self.IdxArray = []
                    for i in range(self.count):
                        self.IdxArray.append(np.array(sum(self.Parts[i][1], [])))
                    
                    self.glDraw()
            else:
                Vtx = []
                for i in range(self.count):
                    for j in range(len(self.Parts[i][0])):
                        Vtx += self.Parts[i][0][j]
                        Vtx += self.red(self.T[1][i] - Tmin, Tmax - Tmin)
                self.VtxArray = np.array(Vtx, dtype = np.float32)

                self.IdxArray = []
                for i in range(self.count):
                    self.IdxArray.append(np.array(sum(self.Parts[i][1], [])))
                
                self.glDraw()

    def red(self, brightness, r):
        brightness = brightness / r
        return [0.0 + brightness, 0.0, 1.0 - brightness]

    def setRotX(self, val):
        self.rotX = np.pi * val
        self.glDraw()

    def setRotY(self, val):
        self.rotY = np.pi * val
        self.glDraw()

    def setRotZ(self, val):
        self.rotZ = np.pi * val
        self.glDraw()

        
class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        
        self.resize(900, 900)
        self.setWindowTitle('Task1')

        self.glWidget = GLWidget(self)

        self.initGUI()
        
    def initGUI(self):
        central_widget = QtWidgets.QWidget()
        gui_layout = QtWidgets.QGridLayout()
        central_widget.setLayout(gui_layout)

        self.setCentralWidget(central_widget)

        gui_layout.addWidget(self.glWidget, 0, 0, 1, 0)

        self.Max = QtWidgets.QLabel('Maximum temperature: ')
        self.Max.setFixedSize(450, 10)
        gui_layout.addWidget(self.Max, 1, 0)

        self.Tmax = QtWidgets.QLineEdit('64')
        gui_layout.addWidget(self.Tmax, 1, 1)

        self.Min = QtWidgets.QLabel('Minimum temperature: ')
        self.Min.setFixedSize(450, 10)
        gui_layout.addWidget(self.Min, 2, 0)

        self.Tmin = QtWidgets.QLineEdit('60')
        gui_layout.addWidget(self.Tmin, 2, 1)

        sliderX = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        sliderX.valueChanged.connect(lambda val: self.glWidget.setRotX(val))

        sliderY = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        sliderY.valueChanged.connect(lambda val: self.glWidget.setRotY(val))

        sliderZ = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        sliderZ.valueChanged.connect(lambda val: self.glWidget.setRotZ(val))
        
        gui_layout.addWidget(sliderX, 3, 0, 1, 0)
        gui_layout.addWidget(sliderY, 4, 0, 1, 0)
        gui_layout.addWidget(sliderZ, 5, 0, 1, 0)

        self.Mbtn = QtWidgets.QPushButton("Model file")
        self.Mbtn.clicked.connect(self.getMfile)
        gui_layout.addWidget(self.Mbtn, 6, 0)

        self.Mname = QtWidgets.QLabel('E:/Python/Задание1/model1.obj')
        gui_layout.addWidget(self.Mname, 6, 1)

        self.Pbtn = QtWidgets.QPushButton("Parameter file")
        self.Pbtn.clicked.connect(self.getPfile)
        gui_layout.addWidget(self.Pbtn, 7, 0)

        self.Pname = QtWidgets.QLabel('E:/Python/Задание1/parameters.json')
        gui_layout.addWidget(self.Pname, 7, 1)

        self.Tbtn = QtWidgets.QPushButton("Temperature file")
        self.Tbtn.clicked.connect(self.getTfile)
        gui_layout.addWidget(self.Tbtn, 8, 0)

        self.Tname = QtWidgets.QLabel('E:/Python/Задание1/temperature.csv')
        gui_layout.addWidget(self.Tname, 8, 1)

        self.Time_name = QtWidgets.QLabel('Time = ')
        gui_layout.addWidget(self.Time_name, 9, 0)

        self.Time_param = QtWidgets.QLineEdit('120')
        gui_layout.addWidget(self.Time_param, 9, 1)

        self.Start_btn = QtWidgets.QPushButton("Solution start")
        self.Start_btn.clicked.connect(self.Start)
        gui_layout.addWidget(self.Start_btn, 10, 0)

        self.Start_btn = QtWidgets.QPushButton("Plot")
        self.Start_btn.clicked.connect(self.Plot)
        gui_layout.addWidget(self.Start_btn, 10, 1)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.valueChanged.connect(self.start_thread)
        gui_layout.addWidget(self.slider, 11, 0, 1, 0)

        self.Time_ = QtWidgets.QLabel('0')
        self.Time_.setFixedSize(450, 10)
        gui_layout.addWidget(self.Time_, 12, 0)
    
    def start_thread(self):
        self.glWidget.initGeometry(int(self.Tmax.text()), int(self.Tmin.text()))

    def Plot(self):
        self.glWidget.Plot_sol()

    def Start(self):
        self.slider.setRange(0, int(self.Time_param.text()) - 1)

        self.time_th = TimeThread(mainwindow=self)
        self.time_th.start()

    def getMfile(self):
        fname = QtWidgets.QFileDialog.getOpenFileName()[0]
        self.Mname.setText(fname)

    def getPfile(self):
        fname = QtWidgets.QFileDialog.getOpenFileName()[0]
        self.Pname.setText(fname)

    def getTfile(self):
        fname = QtWidgets.QFileDialog.getOpenFileName()[0]
        self.Tname.setText(fname)

        
if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    
    win = MainWindow()
    win.show()

    sys.exit(app.exec_())