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

from PyQt5.QtCore import QThread
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
from multiprocessing import Process

import sys
import time

import A_cyth

import pyopencl as cl

cntxt = cl.create_some_context()
queue = cl.CommandQueue(cntxt)

class Verlet_Thread(QThread):
    def __init__(self, mainGLW, pl_i, parent=None):
        super().__init__()
        self.mainGLW = mainGLW
        self.pl_i = pl_i

    def run(self):
        for j in range(len(self.mainGLW.pos_x)):
            if j != self.pl_i and self.pl_i != 0:
                self.mainGLW.a[self.pl_i][0] += self.mainGLW.G * self.mainGLW.mas[j] * (self.mainGLW.pos_x[j] - self.mainGLW.pos_x[self.pl_i]) / \
                                        (math.sqrt( (self.mainGLW.pos_x[j] - self.mainGLW.pos_x[self.pl_i])**2 + (self.mainGLW.pos_y[j] - self.mainGLW.pos_y[self.pl_i])**2 ))**3
                
                self.mainGLW.a[self.pl_i][1] += self.mainGLW.G * self.mainGLW.mas[j] * (self.mainGLW.pos_y[j] - self.mainGLW.pos_y[self.pl_i]) / \
                                        (math.sqrt( (self.mainGLW.pos_x[j] - self.mainGLW.pos_x[self.pl_i])**2 + (self.mainGLW.pos_y[j] - self.mainGLW.pos_y[self.pl_i])**2 ))**3
                    



class GLWidget(QtOpenGL.QGLWidget):
    def __init__(self, parent=None):
        self.parent = parent
        QtOpenGL.QGLWidget.__init__(self, parent)
            
    def initializeGL(self):
        self.qglClearColor(QtGui.QColor(255, 255, 255))
        # self.qglClearColor(QtGui.QColor(0, 0, 0))
        gl.glEnable(gl.GL_DEPTH_TEST)

        self.scale = 1.0

        self.radio = [1.0, 0.244, 0.6052, 0.6371, 0.339, 6.9911, 5.8232, 2.5362, 2.4622]
        self.color = [[1.0, 0.6, 0.0], [1.0, 0.8, 0.4], [1.0, 0.8, 0.0], [0.0, 0.0, 1.0], [0.8, 0.2, 0.2], [1.0, 0.6, 0.2], [1.0, 0.8, 0.6], [0.6, 0.8, 1.0], [0.0, 0.4, 0.8]]

        self.G = 6.67 / 10**11

        self.mas = [1989000, 0.32868, 4.81068, 5.97600, 0.63345, 1876.64328, 561.80376, 86.05440, 101.59200]
        self.mas = [i * 10**4 for i in self.mas]

        self.num_elem = 9

        self.initialize()

    def initialize(self):
        self.t = int(self.parent.Time_param.text())
        self.delt_t = float(self.parent.Step_param.text())
        self.time = np.linspace(0, self.t, int(self.t / self.delt_t))

        # уменьшение в 10**10
        self.pos_x = [0, 5.7910006, 10.8199995, 14.9599951, 22.793992, 77.8330257, 142.9400028, 287.0989228, 450.4299579]
        self.pos_y = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.v_x = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.v_y = [0, 47.87, 35.02, 29.78, 24.13, 13.07, 9.69, 6.81, 5.43]
        self.v_y = [i / 10**2 for i in self.v_y]

        self.num_elem = len(self.pos_x)

        self.a = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
        for i in range(self.num_elem):
            for j in range(self.num_elem):
                if j != i and i != 0:
                    self.a[i][0] += self.G * self.mas[j] * (self.pos_x[j] - self.pos_x[i]) / \
                                                                (math.sqrt( (self.pos_x[j] - self.pos_x[i])**2 + (self.pos_y[j] - self.pos_y[i])**2 ))**3

                    self.a[i][1] += self.G * self.mas[j] * (self.pos_y[j] - self.pos_y[i]) / \
                                                                (math.sqrt( (self.pos_x[j] - self.pos_x[i])**2 + (self.pos_y[j] - self.pos_y[i])**2 ))**3

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

        for pl in range(self.num_elem):
            gl.glBegin(gl.GL_POLYGON)
            gl.glColor3f(self.color[pl][0], self.color[pl][1], self.color[pl][2])
            for i in range(360):
                theta = i*math.pi/180
                gl.glVertex2f(self.pos_x[pl] + self.radio[pl]*math.cos(theta), self.pos_y[pl] + self.radio[pl]*math.sin(theta))
            gl.glEnd()

        gl.glPopMatrix()

    def wheelEvent(self, pe):
        if pe.angleDelta().y() > 0:
            self.scale *= 1.1
        elif pe.angleDelta().y() < 0:
            self.scale /= 1.1
    
        self.glDraw()

    def _ODE(self, r_v, t):
        dr_vdt = []

        c = int(len(r_v) / 4)

        for i in range(2*c):
            dr_vdt.append(r_v[2*c + i])

        for i in range(c):
            a_i = 0
            for j in range(c):
                if j != i and i != 0:
                    a_i += self.G * self.mas[j] * (r_v[j] - r_v[i]) / \
                                                            (math.sqrt( (r_v[j] - r_v[i])**2 + (r_v[c + j] - r_v[c + i])**2 ))**3

            dr_vdt.append(a_i)

        for i in range(c):
            a_i = 0
            for j in range(c):
                if j != i and i != 0:
                    a_i += self.G * self.mas[j] * (r_v[c + j] - r_v[c + i]) / \
                                                            (math.sqrt( (r_v[j] - r_v[i])**2 + (r_v[c + j] - r_v[c + i])**2 ))**3

            dr_vdt.append(a_i)
        
        return dr_vdt

    def Ode(self):
        # self.pos = [[0, 0], [5.7910006, 0], [10.8199995, 0], [14.9599951, 0], [22.793992, 0], [77.8330257, 0], [142.9400028, 0], [287.0989228, 0], [450.4299579, 0]]
        # self.v = [[0, 0], [0, 47.87], [0, 35.02], [0, 29.78], [0, 24.13], [0, 13.07], [0, 9.69], [0, 6.81], [0, 5.43]]
        # for j in range(len(self.v)):
        #     self.v[j] = [i / 10**2 for i in self.v[j]]

        self.initialize()

        r_v_x_y0 = []
        for i in range(self.num_elem):
            r_v_x_y0.append(self.pos_x[i])
        for i in range(self.num_elem):
            r_v_x_y0.append(self.pos_y[i])
        for i in range(self.num_elem):
            r_v_x_y0.append(self.v_x[i])
        for i in range(self.num_elem):
            r_v_x_y0.append(self.v_y[i])

        self.r_v_x_y = odeint(self._ODE, r_v_x_y0, self.time)
        
        for t_i in range(len(self.time)):
            for i in range(self.num_elem):
                self.pos_x[i] = self.r_v_x_y[t_i][i]
                self.pos_y[i] = self.r_v_x_y[t_i][self.num_elem + i]
            for i in range(self.num_elem):
                self.v_x[i] = self.r_v_x_y[t_i][2*self.num_elem + i]
                self.v_y[i] = self.r_v_x_y[t_i][2*self.num_elem + self.num_elem + i]

            # self.glDraw()

        pos_x = []
        pos_y = []
        speed_x = []
        speed_y = []
        for t_i in range(len(self.time)):
            pos_x.append(self.r_v_x_y[t_i][0:self.num_elem+1])
            pos_y.append(self.r_v_x_y[t_i][self.num_elem+1:2*self.num_elem+1])
            speed_x.append(self.r_v_x_y[t_i][2*self.num_elem+1:3*self.num_elem+1])
            speed_y.append(self.r_v_x_y[t_i][3*self.num_elem+1:4*self.num_elem+1])

        self.Sol_ode = np.concatenate((pos_x, pos_y, speed_x, speed_y), axis=-1)

    def Verlet(self):
        start_time = time.time()

        self.initialize()

        self.Sol_verl = []

        pos_x_buf = []
        pos_y_buf = []
        speed_x_buf = []
        speed_y_buf = []

        self.pos_x_i = []
        self.pos_y_i = []
        self.speed_x_i = []
        self.speed_y_i = []

        for t_i in range(len(self.time)):
            for i in range(self.num_elem):
                pos_x_buf.append(self.pos_x[i])
                pos_y_buf.append(self.pos_y[i])
                speed_x_buf.append(self.v_x[i])
                speed_y_buf.append(self.v_y[i])
            self.pos_x_i.append(pos_x_buf)
            self.pos_y_i.append(pos_y_buf)
            self.speed_x_i.append(speed_x_buf)
            self.speed_y_i.append(speed_y_buf)

            pos_x_buf = []
            pos_y_buf = []
            speed_x_buf = []
            speed_y_buf = []

            for i in range(self.num_elem):
                self.pos_x[i] = self.pos_x[i] + self.v_x[i] * self.delt_t + 0.5 * self.a[i][0] * self.delt_t**2
                self.pos_y[i] = self.pos_y[i] + self.v_y[i] * self.delt_t + 0.5 * self.a[i][1] * self.delt_t**2

            a_old = self.a
            self.a = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
            for i in range(self.num_elem):
                for j in range(self.num_elem):
                    if j != i and i != 0:
                        self.a[i][0] += self.G * self.mas[j] * (self.pos_x[j] - self.pos_x[i]) / \
                                                                (math.sqrt( (self.pos_x[j] - self.pos_x[i])**2 + (self.pos_y[j] - self.pos_y[i])**2 ))**3

                        self.a[i][1] += self.G * self.mas[j] * (self.pos_y[j] - self.pos_y[i]) / \
                                                                (math.sqrt( (self.pos_x[j] - self.pos_x[i])**2 + (self.pos_y[j] - self.pos_y[i])**2 ))**3
            if t_i == 1:
                print(self.a)
            for i in range(self.num_elem):
                self.v_x[i] = self.v_x[i] + 0.5 * (self.a[i][0] + a_old[i][0]) * self.delt_t
                self.v_y[i] = self.v_y[i] + 0.5 * (self.a[i][1] + a_old[i][1]) * self.delt_t
            
            # self.glDraw()
            # time.sleep(0.1)
        self.Sol_verl = np.concatenate((self.pos_x_i, self.pos_y_i, self.speed_x_i, self.speed_y_i), axis=-1)

        end_time = time.time()
        print(end_time - start_time)

#------------------------------------------------------------------

    def V_th(self, v_x, v_y, a, a_old, delt_t, i):
        v_out_x = v_x[i] + 0.5 * (a[i][0] + a_old[i][0]) * delt_t
        v_out_y = v_y[i] + 0.5 * (a[i][1] + a_old[i][1]) * delt_t
        return v_out_x, v_out_y, i

    def Verlet_thr(self):
        start_time = time.time()

        self.initialize()

        self.Sol_verl_thr = []

        pos_x_buf = []
        pos_y_buf = []
        speed_x_buf = []
        speed_y_buf = []

        self.pos_x_i = []
        self.pos_y_i = []
        self.speed_x_i = []
        self.speed_y_i = []

        for t_i in range(len(self.time)):
            for i in range(self.num_elem):
                pos_x_buf.append(self.pos_x[i])
                pos_y_buf.append(self.pos_y[i])
                speed_x_buf.append(self.v_x[i])
                speed_y_buf.append(self.v_y[i])
            self.pos_x_i.append(pos_x_buf)
            self.pos_y_i.append(pos_y_buf)
            self.speed_x_i.append(speed_x_buf)
            self.speed_y_i.append(speed_y_buf)

            pos_x_buf = []
            pos_y_buf = []
            speed_x_buf = []
            speed_y_buf = []

            # threads = []

            for i in range(self.num_elem):
                self.pos_x[i] = self.pos_x[i] + self.v_x[i] * self.delt_t + 0.5 * self.a[i][0] * self.delt_t**2
                self.pos_y[i] = self.pos_y[i] + self.v_y[i] * self.delt_t + 0.5 * self.a[i][1] * self.delt_t**2

            a_old = self.a
            self.a = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]

            with ThreadPoolExecutor(max_workers=self.num_elem) as executor:
                jobs = []
                for i in range(self.num_elem):
                    jobs.append(executor.submit(A_th, pos_x = self.pos_x, pos_y = self.pos_y, G = self.G, mas = self.mas, pl_i = i ))
                
                for job in as_completed(jobs):
                    result_done = job.result()
                    i = result_done[-1]
                    self.a[i][0] = result_done[0]
                    self.a[i][1] = result_done[1]

            # for i in range(self.num_elem):
            #     t = threading.Thread(target = A_th, args = (self.pos_x, self.pos_y, self.G, self.mas, i, self.a[i]))
            #     t.setDaemon(True)
            #     t.start()
            # main_thread = threading.currentThread()
            # for t in threading.enumerate():
            #     if t is main_thread:
            #         continue
            #     t.join()

            # for i in range(len(self.pos_x)):
            #     Verlet_th = Verlet_Thread(mainGLW=self, pl_i=i)
            #     threads.append(Verlet_th)
            #     Verlet_th.start()

            # for th in threads:
            #     th.wait()


            # with ThreadPoolExecutor(max_workers=len(self.pos_x)) as executor:
            #     jobs = []
            #     for i in range(len(self.pos_x)):
            #         jobs.append(executor.submit(self.V_th, v_x = self.v_x, v_y = self.v_y, a = self.a, a_old = a_old, delt_t = self.delt_t, i = i ))
                
            #     for job in as_completed(jobs):
            #         result_done = job.result()
            #         i = result_done[-1]
            #         self.v_x[i] = result_done[0]
            #         self.v_y[i] = result_done[1]

            for i in range(self.num_elem):
                self.v_x[i] = self.v_x[i] + 0.5 * (self.a[i][0] + a_old[i][0]) * self.delt_t
                self.v_y[i] = self.v_y[i] + 0.5 * (self.a[i][1] + a_old[i][1]) * self.delt_t

            # self.glDraw()

        self.Sol_verl_thr = np.concatenate((self.pos_x_i, self.pos_y_i, self.speed_x_i, self.speed_y_i), axis=-1)

        end_time = time.time()
        print(end_time - start_time)

#------------------------------------------------------------------

    def Verlet_proc(self):
        start_time = time.time()

        self.initialize()

        self.Sol_verl_proc = []

        pos_x_buf = []
        pos_y_buf = []
        speed_x_buf = []
        speed_y_buf = []

        self.pos_x_i = []
        self.pos_y_i = []
        self.speed_x_i = []
        self.speed_y_i = []

        with ProcessPoolExecutor(max_workers = 4) as executor:
            for t_i in range(len(self.time)):
                for i in range(self.num_elem):
                    pos_x_buf.append(self.pos_x[i])
                    pos_y_buf.append(self.pos_y[i])
                    speed_x_buf.append(self.v_x[i])
                    speed_y_buf.append(self.v_y[i])
                self.pos_x_i.append(pos_x_buf)
                self.pos_y_i.append(pos_y_buf)
                self.speed_x_i.append(speed_x_buf)
                self.speed_y_i.append(speed_y_buf)

                pos_x_buf = []
                pos_y_buf = []
                speed_x_buf = []
                speed_y_buf = []

                # threads = []

                for i in range(self.num_elem):
                    self.pos_x[i] = self.pos_x[i] + self.v_x[i] * self.delt_t + 0.5 * self.a[i][0] * self.delt_t**2
                    self.pos_y[i] = self.pos_y[i] + self.v_y[i] * self.delt_t + 0.5 * self.a[i][1] * self.delt_t**2

                a_old = self.a
                self.a = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]

                # with ProcessPoolExecutor(max_workers=self.num_elem) as executor:
                jobs = []
                for i in range(self.num_elem):
                    jobs.append(executor.submit(A_th, pos_x = self.pos_x, pos_y = self.pos_y, G = self.G, mas = self.mas, pl_i = i ))
                
                for job in as_completed(jobs):
                    result_done = job.result()
                    i = result_done[-1]
                    self.a[i][0] = result_done[0]
                    self.a[i][1] = result_done[1]

                

                for i in range(self.num_elem):
                    self.v_x[i] = self.v_x[i] + 0.5 * (self.a[i][0] + a_old[i][0]) * self.delt_t
                    self.v_y[i] = self.v_y[i] + 0.5 * (self.a[i][1] + a_old[i][1]) * self.delt_t

                # self.glDraw()

        self.Sol_verl_proc = np.concatenate((self.pos_x_i, self.pos_y_i, self.speed_x_i, self.speed_y_i), axis=-1)

        end_time = time.time()
        print(end_time - start_time)

#------------------------------------------------------------------

    def Verlet_Cyth(self):
        start_time = time.time()

        self.initialize()

        self.Sol_verl_cyth = []

        pos_x_buf = []
        pos_y_buf = []
        speed_x_buf = []
        speed_y_buf = []

        self.pos_x_i = []
        self.pos_y_i = []
        self.speed_x_i = []
        self.speed_y_i = []

        for t_i in range(len(self.time)):
            for i in range(self.num_elem):
                pos_x_buf.append(self.pos_x[i])
                pos_y_buf.append(self.pos_y[i])
                speed_x_buf.append(self.v_x[i])
                speed_y_buf.append(self.v_y[i])
            self.pos_x_i.append(pos_x_buf)
            self.pos_y_i.append(pos_y_buf)
            self.speed_x_i.append(speed_x_buf)
            self.speed_y_i.append(speed_y_buf)

            pos_x_buf = []
            pos_y_buf = []
            speed_x_buf = []
            speed_y_buf = []

            # threads = []

            for i in range(self.num_elem):
                self.pos_x[i] = self.pos_x[i] + self.v_x[i] * self.delt_t + 0.5 * self.a[i][0] * self.delt_t**2
                self.pos_y[i] = self.pos_y[i] + self.v_y[i] * self.delt_t + 0.5 * self.a[i][1] * self.delt_t**2

            a_old = self.a
            self.a = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
            for i in range(self.num_elem):
                res = A_cyth.A_th(self.pos_x, self.pos_y, self.G, self.mas, i)

                self.a[i][0] = res[0]
                self.a[i][1] = res[1]

            for i in range(self.num_elem):
                self.v_x[i] = self.v_x[i] + 0.5 * (self.a[i][0] + a_old[i][0]) * self.delt_t
                self.v_y[i] = self.v_y[i] + 0.5 * (self.a[i][1] + a_old[i][1]) * self.delt_t

            # self.glDraw()

        self.Sol_verl_cyth = np.concatenate((self.pos_x_i, self.pos_y_i, self.speed_x_i, self.speed_y_i), axis=-1)

        end_time = time.time()
        print(end_time - start_time)

#------------------------------------------------------------------

    def Verlet_CL(self):
        start_time = time.time()

        self.initialize()
        
        self.Sol_verl_cl = []

        pos_x_buf0 = []
        pos_y_buf0 = []
        speed_x_buf0 = []
        speed_y_buf0 = []

        self.pos_x_i = []
        self.pos_y_i = []
        self.speed_x_i = []
        self.speed_y_i = []

        # cntxt = cl.create_some_context()
        # queue = cl.CommandQueue(cntxt)

        # code = """
        # __kernel void A_cl(__global float* pos_x, __global float* pos_y, __global float* param, __global float* mas, __global float* res) 
        # {
        #     float a_x = 0;
        #     float a_y = 0;
        #     float G = param[0];
        #     int pl_i = param[1];
        #     int size = param[2];
        #     for (int j = 0; j < size; j++)
        #     {
        #         if (j != pl_i && pl_i != 0)
        #         {
        #             a_x += G * mas[j] * (pos_x[j] - pos_x[pl_i]) / 
        #                 ( (sqrt( (pos_x[j] - pos_x[pl_i])*(pos_x[j] - pos_x[pl_i]) + (pos_y[j] - pos_y[pl_i])*(pos_y[j] - pos_y[pl_i]) ))*
        #                 (sqrt( (pos_x[j] - pos_x[pl_i])*(pos_x[j] - pos_x[pl_i]) + (pos_y[j] - pos_y[pl_i])*(pos_y[j] - pos_y[pl_i]) ))*
        #                 (sqrt( (pos_x[j] - pos_x[pl_i])*(pos_x[j] - pos_x[pl_i]) + (pos_y[j] - pos_y[pl_i])*(pos_y[j] - pos_y[pl_i]) )) );
                    
        #             a_y += G * mas[j] * (pos_y[j] - pos_y[pl_i]) / 
        #                 ( (sqrt( (pos_x[j] - pos_x[pl_i])*(pos_x[j] - pos_x[pl_i]) + (pos_y[j] - pos_y[pl_i])*(pos_y[j] - pos_y[pl_i]) ))*
        #                 (sqrt( (pos_x[j] - pos_x[pl_i])*(pos_x[j] - pos_x[pl_i]) + (pos_y[j] - pos_y[pl_i])*(pos_y[j] - pos_y[pl_i]) ))*
        #                 (sqrt( (pos_x[j] - pos_x[pl_i])*(pos_x[j] - pos_x[pl_i]) + (pos_y[j] - pos_y[pl_i])*(pos_y[j] - pos_y[pl_i]) )) );
        #         }
        #     }
        #     res[0] = a_x;
        #     res[1] = a_y;
        #     res[2] = pl_i;
        # }
        # """

        code = """
        __kernel void A_cl(__global float* pos_x, __global float* pos_y, __global float* param, __global float* mas, __global float* res, __const float G, __const int pl_i, __const int size) 
        {
            int gid = get_global_id(0);
            res[gid] = G;
            res[gid] = pl_i;
        }
        """

        bld = cl.Program(cntxt, code).build()

        for t_i in range(len(self.time)):
            for i in range(self.num_elem):
                pos_x_buf0.append(self.pos_x[i])
                pos_y_buf0.append(self.pos_y[i])
                speed_x_buf0.append(self.v_x[i])
                speed_y_buf0.append(self.v_y[i])
            self.pos_x_i.append(pos_x_buf0)
            self.pos_y_i.append(pos_y_buf0)
            self.speed_x_i.append(speed_x_buf0)
            self.speed_y_i.append(speed_y_buf0)

            pos_x_buf0 = []
            pos_y_buf0 = []
            speed_x_buf0 = []
            speed_y_buf0 = []

            # threads = []

            for i in range(self.num_elem):
                self.pos_x[i] = self.pos_x[i] + self.v_x[i] * self.delt_t + 0.5 * self.a[i][0] * self.delt_t**2
                self.pos_y[i] = self.pos_y[i] + self.v_y[i] * self.delt_t + 0.5 * self.a[i][1] * self.delt_t**2

            a_old = self.a
            self.a = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
            for i in range(self.num_elem):
                param = np.array([self.G, i, self.num_elem])

                pos_x_buf = cl.Buffer(cntxt, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = np.array(self.pos_x))
                pos_y_buf = cl.Buffer(cntxt, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = np.array(self.pos_y))
                # G_buf = cl.Buffer(cntxt, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = self.G)
                mas_buf = cl.Buffer(cntxt, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = np.array(self.mas))
                # i_buf = cl.Buffer(cntxt, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = i)
                # size_buf = cl.Buffer(cntxt, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = self.num_elem)
                param_buf = cl.Buffer(cntxt, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = np.array(param))

                # res = np.ones(3)
                # res_buf = cl.Buffer(cntxt, cl.mem_flags.WRITE_ONLY, res.nbytes)
                res_buf = cl.Buffer(cntxt, cl.mem_flags.WRITE_ONLY, np.array(self.pos_x).nbytes)
                
                # launch = bld.A_cl(queue, np.array(self.pos_x).shape, None, pos_x_buf, pos_y_buf, param_buf, mas_buf, res_buf)
                # launch.wait()
                bld.A_cl(queue, np.array(self.pos_x).shape, None, pos_x_buf, pos_y_buf, param_buf, mas_buf, res_buf, np.float32(self.G), np.int32(i), np.int32(self.num_elem))
                res = np.empty_like(np.array(self.pos_x))
                cl.enqueue_copy(queue, res, res_buf)
                
                self.a[i][0] = res[0]
                self.a[i][1] = res[1]
            if t_i == 1:
                print(self.a)

            for i in range(self.num_elem):
                self.v_x[i] = self.v_x[i] + 0.5 * (self.a[i][0] + a_old[i][0]) * self.delt_t
                self.v_y[i] = self.v_y[i] + 0.5 * (self.a[i][1] + a_old[i][1]) * self.delt_t

            # self.glDraw()

        self.Sol_verl_cl = np.concatenate((self.pos_x_i, self.pos_y_i, self.speed_x_i, self.speed_y_i), axis=-1)

        end_time = time.time()
        print(end_time - start_time)

#------------------------------------------------------------------

    def Plot_dif(self):
        plt.plot(np.sqrt( np.sum((self.Sol_ode - self.Sol_verl)**2, axis=-1) ), 'b', label='Error Verlet')
        # plt.plot(np.sqrt( np.sum((self.Sol_ode - self.Sol_verl_thr)**2, axis=-1) ), 'g', label='Error Verlet with threading')
        # plt.plot(np.sqrt( np.sum((self.Sol_ode - self.Sol_verl_proc)**2, axis=-1) ), 'r', label='Error Verlet with process')
        # plt.plot(np.sqrt( np.sum((self.Sol_ode - self.Sol_verl_cyth)**2, axis=-1) ), 'c', label='Error Verlet with Cython')
        plt.plot(np.sqrt( np.sum((self.Sol_ode - self.Sol_verl_cl)**2, axis=-1) ), 'm', label='Error Verlet with OpenCL')
        plt.legend(loc='best')
        plt.xlabel('t')
        plt.grid()
        plt.show()

#------------------------------------------------------------------
def A_th(pos_x, pos_y, G, mas, pl_i):
        a_x = 0
        a_y = 0
        for j in range(len(pos_x)):
            if j != pl_i and pl_i != 0:
                a_x += G * mas[j] * (pos_x[j] - pos_x[pl_i]) / \
                                        (math.sqrt( (pos_x[j] - pos_x[pl_i])**2 + (pos_y[j] - pos_y[pl_i])**2 ))**3
                
                a_y += G * mas[j] * (pos_y[j] - pos_y[pl_i]) / \
                                        (math.sqrt( (pos_x[j] - pos_x[pl_i])**2 + (pos_y[j] - pos_y[pl_i])**2 ))**3
        return a_x, a_y, pl_i

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

        self.glWidget.setFixedSize(900, 700)
        gui_layout.addWidget(self.glWidget, 0, 0, 1, 0)

        self.Time_name = QtWidgets.QLabel('Time = ')
        self.Time_name.setFixedSize(450, 10)
        gui_layout.addWidget(self.Time_name, 1, 0)

        self.Time_param = QtWidgets.QLineEdit('100')
        gui_layout.addWidget(self.Time_param, 1, 1)

        self.Step_name = QtWidgets.QLabel('Step = ')
        self.Step_name.setFixedSize(450, 10)
        gui_layout.addWidget(self.Step_name, 2, 0)

        self.Step_param = QtWidgets.QLineEdit('0.1')
        gui_layout.addWidget(self.Step_param, 2, 1)

        self.Ode_btn = QtWidgets.QPushButton("Odeint")
        self.Ode_btn.clicked.connect(self.Widget_Ode)
        gui_layout.addWidget(self.Ode_btn, 3, 0)

        self.Verl_btn = QtWidgets.QPushButton("Verlet method")
        self.Verl_btn.clicked.connect(self.Widget_Verlet)
        gui_layout.addWidget(self.Verl_btn, 3, 1)

        self.Verl_thr_btn = QtWidgets.QPushButton("Verlet method with threading")
        self.Verl_thr_btn.clicked.connect(self.Widget_Verlet_thr)
        gui_layout.addWidget(self.Verl_thr_btn, 4, 0)

        self.Verl_proc_btn = QtWidgets.QPushButton("Verlet method with process")
        self.Verl_proc_btn.clicked.connect(self.Widget_Verlet_proc)
        gui_layout.addWidget(self.Verl_proc_btn, 4, 1)

        self.Verl_Cyth_btn = QtWidgets.QPushButton("Verlet method - Cython")
        self.Verl_Cyth_btn.clicked.connect(self.Widget_Verlet_Cyth)
        gui_layout.addWidget(self.Verl_Cyth_btn, 5, 0)

        self.Verl_CL_btn = QtWidgets.QPushButton("Verlet method - OpenCL")
        self.Verl_CL_btn.clicked.connect(self.Widget_Verlet_CL)
        gui_layout.addWidget(self.Verl_CL_btn, 5, 1)

        self.Plot_btn = QtWidgets.QPushButton("Plot")
        self.Plot_btn.clicked.connect(self.Plot)
        gui_layout.addWidget(self.Plot_btn, 6, 0, 1, 0)

    def Widget_Ode(self):
        self.glWidget.Ode()

    def Widget_Verlet(self):
        self.glWidget.Verlet()

    def Widget_Verlet_thr(self):
        self.glWidget.Verlet_thr()

    def Widget_Verlet_proc(self):
        self.glWidget.Verlet_proc()

    def Widget_Verlet_Cyth(self):
        self.glWidget.Verlet_Cyth()

    def Widget_Verlet_CL(self):
        self.glWidget.Verlet_CL()

    def Plot(self):
        self.glWidget.Plot_dif()



if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    
    win = MainWindow()
    win.show()

    sys.exit(app.exec_())