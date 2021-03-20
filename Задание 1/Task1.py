import numpy as np
import math
import json
import OpenGL
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtOpenGL

import OpenGL.GL as gl
from OpenGL import GLU
from OpenGL.arrays import vbo

from scipy.integrate import odeint
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

import sys

class GLWidget(QtOpenGL.QGLWidget):
    def __init__(self, parent=None):
        self.parent = parent
        QtOpenGL.QGLWidget.__init__(self, parent)
            
    def initializeGL(self):
        self.qglClearColor(QtGui.QColor(255, 255, 255))
        gl.glEnable(gl.GL_DEPTH_TEST)

        self.Parts = []
        self.T = []

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
            gl.glVertexPointer(3, gl.GL_FLOAT, 0, self.vertVBO)
            # gl.glColorPointer(3, gl.GL_FLOAT, 0, self.colorVBO)

            for i in range(self.count):
                gl.glDrawElements(gl.GL_TRIANGLES, len(self.cubeIdxArray[i]), gl.GL_UNSIGNED_INT, self.cubeIdxArray[i])

        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        gl.glDisableClientState(gl.GL_COLOR_ARRAY)

        gl.glPopMatrix()

    def Parsing(self, Mname, Pname, time):
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
                    # err = int(values[1])
                    err = 1
                for i in range(len(values) - 1):
                    values[i + 1] = int(values[i + 1]) - err
                self.Parts[index][1].append(list(map(int, values[1:4])))

        S_ij = np.zeros((self.count, self.count))
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
                    S_ij[p][k] = S_k
                else:
                    S_ij[p][k] = S_p
        
        with open(Pname, 'r') as f:
            param = json.load(f)

        C = []
        for c_i in param['c']:
            C.append(c_i['c_i'])

        Lamb = []
        for lamb_i in param['lamb']:
            Lamb.append(lamb_i['lamb_i'])

        E = []
        for e_i in param['e']:
            E.append(e_i['e_i'])

        Q = []
        for Q_i in param['Q']:
            Q.append(Q_i['Q_i'])

        self.T = self.Solve(S_ij, C, Lamb, E, Q, time)

        self.initGeometry()

    def Solve(self, S_ij, C, Lamb, E, Q, time):
        self.t = np.linspace(0, time, 100)

        T0 = fsolve(self.Stationary_solution, np.zeros(self.count), args=(0.0, S_ij, C, Lamb, E, Q))
        
        # print(T0)

        self.T = odeint(self.ODE, T0, self.t, args=(S_ij, C, Lamb, E,  Q))

        print(self.T)

        return self.T

    def Plot_sol(self):
        if len(self.T) != 0:
            plt.plot(self.t, self.T[:, 0], 'b', label='Part1')
            plt.plot(self.t, self.T[:, 1], 'g', label='Part2')
            plt.plot(self.t, self.T[:, 2], 'r', label='Part3')
            plt.plot(self.t, self.T[:, 3], 'c', label='Part4')
            plt.plot(self.t, self.T[:, 4], 'm', label='Part5')
            plt.legend(loc='best')
            plt.xlabel('t')
            plt.grid()
            plt.show()

    def Stationary_solution(self, T, t, S_ij, C, Lamb, E, Q):
        C0 = 5.67
        return [(-Lamb[0] * S_ij[0][1] * (T[1] - T[0]) - E[0] * S_ij[0][0] * C0 * (T[0]/100)**4) / C[0],
                (-Lamb[1] * S_ij[1][2] * (T[2] - T[1]) - Lamb[0] * S_ij[1][0] * (T[0] - T[1]) - E[1] * S_ij[1][1] * C0 * (T[1]/100)**4 + Q[1] * (20 + 3 * math.cos(t/4))) / C[1],
                (-Lamb[2] * S_ij[2][3] * (T[3] - T[2]) - Lamb[1] * S_ij[2][1] * (T[1] - T[2]) - E[2] * S_ij[2][2] * C0 * (T[2]/100)**4) / C[2],
                (-Lamb[3] * S_ij[3][4] * (T[4] - T[3]) - Lamb[2] * S_ij[3][2] * (T[2] - T[3]) - E[3] * S_ij[3][3] * C0 * (T[3]/100)**4) / C[3],
                (-Lamb[3] * S_ij[4][3] * (T[3] - T[4]) - E[4] * S_ij[4][4] * C0 * (T[4]/100)**4) / C[4]]

    def ODE(self, T, t, S_ij, C, Lamb, E, Q):
        T1, T2, T3, T4, T5 = T
        C0 = 5.67
        dTdt = [(-Lamb[0] * S_ij[0][1] * (T2 - T1) - E[0] * S_ij[0][0] * C0 * (T1/100)**4) / C[0],
                (-Lamb[1] * S_ij[1][2] * (T3 - T2) - Lamb[0] * S_ij[1][0] * (T1 - T2) - E[1] * S_ij[1][1] * C0 * (T2/100)**4 + Q[1] * (20 + 3 * math.cos(t/4))) / C[1],
                (-Lamb[2] * S_ij[2][3] * (T4 - T3) - Lamb[1] * S_ij[2][1] * (T2 - T3) - E[2] * S_ij[2][2] * C0 * (T3/100)**4) / C[2],
                (-Lamb[3] * S_ij[3][4] * (T5 - T4) - Lamb[2] * S_ij[3][2] * (T3 - T4) - E[3] * S_ij[3][3] * C0 * (T4/100)**4) / C[3],
                (-Lamb[3] * S_ij[4][3] * (T4 - T5) - E[4] * S_ij[4][4] * C0 * (T5/100)**4) / C[4]]
        return dTdt

    def initGeometry(self):
        if len(self.Parts) != 0:
            Vtx = []
            for i in range(self.count):
                Vtx += self.Parts[i][0]
            self.cubeVtxArray = np.array(Vtx)
            self.vertVBO = vbo.VBO(np.reshape(self.cubeVtxArray, (1, -1)).astype(np.float32))
            self.vertVBO.bind()

            self.cubeIdxArray = []
            for i in range(self.count):
                self.cubeIdxArray.append(np.array(sum(self.Parts[i][1], [])))

    def setRotX(self, val):
        self.rotX = np.pi * val

    def setRotY(self, val):
        self.rotY = np.pi * val

    def setRotZ(self, val):
        self.rotZ = np.pi * val

        
class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        
        self.resize(800, 800)
        self.setWindowTitle('Task1')

        self.glWidget = GLWidget(self)
        self.initGUI()
        
        timer = QtCore.QTimer(self)
        timer.setInterval(20)
        timer.timeout.connect(self.glWidget.updateGL)
        timer.start()
        
    def initGUI(self):
        central_widget = QtWidgets.QWidget()
        gui_layout = QtWidgets.QGridLayout()
        central_widget.setLayout(gui_layout)

        self.setCentralWidget(central_widget)

        gui_layout.addWidget(self.glWidget, 0, 0, 1, 0)

        sliderX = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        sliderX.valueChanged.connect(lambda val: self.glWidget.setRotX(val))

        sliderY = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        sliderY.valueChanged.connect(lambda val: self.glWidget.setRotY(val))

        sliderZ = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        sliderZ.valueChanged.connect(lambda val: self.glWidget.setRotZ(val))
        
        gui_layout.addWidget(sliderX, 1, 0, 1, 0)
        gui_layout.addWidget(sliderY, 2, 0, 1, 0)
        gui_layout.addWidget(sliderZ, 3, 0, 1, 0)


        self.Mbtn = QtWidgets.QPushButton("Model file")
        self.Mbtn.clicked.connect(self.getMfile)
        gui_layout.addWidget(self.Mbtn, 4, 0)

        self.Mname = QtWidgets.QLabel('E:/Python/Задание1/model1.obj')
        gui_layout.addWidget(self.Mname, 4, 1)

        self.Pbtn = QtWidgets.QPushButton("Parameter file")
        self.Pbtn.clicked.connect(self.getPfile)
        gui_layout.addWidget(self.Pbtn, 5, 0)

        self.Pname = QtWidgets.QLabel('E:/Python/Задание1/parameters.json')
        gui_layout.addWidget(self.Pname, 5, 1)

        self.Time_name = QtWidgets.QLabel('Time = ')
        gui_layout.addWidget(self.Time_name, 6, 0)

        self.Time_param = QtWidgets.QLineEdit('20')
        gui_layout.addWidget(self.Time_param, 6, 1)

        self.Start_btn = QtWidgets.QPushButton("Solution start")
        self.Start_btn.clicked.connect(self.Start)
        gui_layout.addWidget(self.Start_btn, 7, 0)

        self.Start_btn = QtWidgets.QPushButton("Plot")
        self.Start_btn.clicked.connect(self.Plot)
        gui_layout.addWidget(self.Start_btn, 7, 1)
        
    def Plot(self):
        self.glWidget.Plot_sol()

    def Start(self):
        self.glWidget.Parsing(self.Mname.text(), self.Pname.text(), int(self.Time_param.text()))

    def getMfile(self):
        fname = QtWidgets.QFileDialog.getOpenFileName()[0]
        self.Mname.setText(fname)

    def getPfile(self):
        fname = QtWidgets.QFileDialog.getOpenFileName()[0]
        self.Pname.setText(fname)

        
if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    
    win = MainWindow()
    win.show()

    sys.exit(app.exec_())
