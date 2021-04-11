import numpy as np
import math

import OpenGL
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtOpenGL

import OpenGL.GL as gl
from OpenGL import GLU

from scipy.integrate import odeint

import matplotlib.pyplot as plt

import sys
import time

class GLWidget(QtOpenGL.QGLWidget):
    def __init__(self, parent=None):
        self.parent = parent
        QtOpenGL.QGLWidget.__init__(self, parent)
            
    def initializeGL(self):
        self.qglClearColor(QtGui.QColor(255, 255, 255))
        # self.qglClearColor(QtGui.QColor(0, 0, 0))
        gl.glEnable(gl.GL_DEPTH_TEST)

        self.scale = 1.0

        # уменьшение в 10**10
        self.pos = [[0, 0], [5.7910006, 0], [10.8199995, 0], [14.9599951, 0], [22.793992, 0], [77.8330257, 0], [142.9400028, 0], [287.0989228, 0], [450.4299579, 0]]

        self.radio = [1.0, 0.244, 0.6052, 0.6371, 0.339, 6.9911, 5.8232, 2.5362, 2.4622]
        self.color = [[1.0, 0.6, 0.0], [1.0, 0.8, 0.4], [1.0, 0.8, 0.0], [0.0, 0.0, 1.0], [0.8, 0.2, 0.2], [1.0, 0.6, 0.2], [1.0, 0.8, 0.6], [0.6, 0.8, 1.0], [0.0, 0.4, 0.8]]

        self.G = 6.67 / 10**11

        self.mas = [1989000, 0.32868, 4.81068, 5.97600, 0.63345, 1876.64328, 561.80376, 86.05440, 101.59200]
        self.mas = [i * 10**4 for i in self.mas]

        self.a = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
        for i in range(len(self.pos)):
            for j in range(len(self.pos)):
                if j != i and i != 0:
                    if self.pos[j][0] != self.pos[i][0]:
                        self.a[i][0] += self.G * self.mas[j] * (self.pos[j][0] - self.pos[i][0]) / \
                                                                (math.sqrt( (self.pos[j][0] - self.pos[i][0])**2 + (self.pos[j][1] - self.pos[i][1])**2 ))**3
                    else:
                        self.a[i][0] = 0
                    if self.pos[j][1] != self.pos[i][1]:
                        self.a[i][1] += self.G * self.mas[j] * (self.pos[j][1] - self.pos[i][1]) / \
                                                                (math.sqrt( (self.pos[j][0] - self.pos[i][0])**2 + (self.pos[j][1] - self.pos[i][1])**2 ))**3
                    else:
                        self.a[i][1] = 0

        self.v = [[0, 0], [0, 47.87], [0, 35.02], [0, 29.78], [0, 24.13], [0, 13.07], [0, 9.69], [0, 6.81], [0, 5.43]]
        for j in range(len(self.v)):
            self.v[j] = [i / 10**2 for i in self.v[j]]

        self.t = 1000
        self.delt_t = 1

    def resizeGL(self, width, height):
        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        aspect = width / float(height)
        
        GLU.gluPerspective(45.0, aspect, 1.0, 1000.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)

    def paintGL(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        gl.glPushMatrix()

        gl.glTranslate(0.0, 0.0, -990.0)
        gl.glScale(self.scale, self.scale, self.scale)

        for pl in range(len(self.pos)):
            gl.glBegin(gl.GL_POLYGON)
            gl.glColor3f(self.color[pl][0], self.color[pl][1], self.color[pl][2])
            for i in range(360):
                theta = i*math.pi/180
                gl.glVertex2f(self.pos[pl][0] + self.radio[pl]*math.cos(theta), self.pos[pl][1] + self.radio[pl]*math.sin(theta))
            gl.glEnd()

        gl.glPopMatrix()

    def wheelEvent(self, pe):
        if pe.angleDelta().y() > 0:
            self.scale *= 1.1
        elif pe.angleDelta().y() < 0:
            self.scale /= 1.1
    
        self.glDraw()

    def Solve(self):
        t_i = 1
        while t_i <= self.t:            
            for i in range(len(self.pos)):
                self.pos[i][0] = self.pos[i][0] + self.v[i][0] * self.delt_t + 0.5 * self.a[i][0] * self.delt_t**2
                self.pos[i][1] = self.pos[i][1] + self.v[i][1] * self.delt_t + 0.5 * self.a[i][1] * self.delt_t**2

            a_old = self.a
            self.a = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
            for i in range(len(self.pos)):
                for j in range(len(self.pos)):
                    if j != i and i != 0:
                        if self.pos[j][0] != self.pos[i][0]:
                            self.a[i][0] += self.G * self.mas[j] * (self.pos[j][0] - self.pos[i][0]) / \
                                                                    (math.sqrt( (self.pos[j][0] - self.pos[i][0])**2 + (self.pos[j][1] - self.pos[i][1])**2 ))**3
                        else:
                            self.a[i][0] = 0
                        if self.pos[j][1] != self.pos[i][1]:
                            self.a[i][1] += self.G * self.mas[j] * (self.pos[j][1] - self.pos[i][1]) / \
                                                                    (math.sqrt( (self.pos[j][0] - self.pos[i][0])**2 + (self.pos[j][1] - self.pos[i][1])**2 ))**3
                        else:
                            self.a[i][1] = 0

            for i in range(len(self.pos)):
                self.v[i][0] = self.v[i][0] + 0.5 * (self.a[i][0] + a_old[i][0]) * self.delt_t
                self.v[i][1] = self.v[i][1] + 0.5 * (self.a[i][1] + a_old[i][1]) * self.delt_t

            t_i += self.delt_t

            self.glDraw()
            # time.sleep(0.1)
        # print(self.pos)
            


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        
        self.resize(900, 900)
        self.setWindowTitle('Task2')

        self.glWidget = GLWidget(self)

        self.initGUI()

        # timer = QtCore.QTimer(self)
        # timer.setInterval(20)
        # timer.timeout.connect(self.glWidget.update)
        # timer.start()
        
    def initGUI(self):
        central_widget = QtWidgets.QWidget()
        gui_layout = QtWidgets.QGridLayout()
        central_widget.setLayout(gui_layout)

        self.setCentralWidget(central_widget)

        self.glWidget.setFixedSize(900, 800)
        gui_layout.addWidget(self.glWidget, 0, 0, 1, 0)

        self.Start_btn = QtWidgets.QPushButton("Solution start")
        self.Start_btn.clicked.connect(self.Start)
        gui_layout.addWidget(self.Start_btn, 1, 0)

    def Start(self):
        self.glWidget.Solve()



if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    
    win = MainWindow()
    win.show()

    sys.exit(app.exec_())