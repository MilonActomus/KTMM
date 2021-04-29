import numpy as np
import math

import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import Process, Queue

import sys
import time

import A_cyth

import pyopencl as cl

import cProfile
import pstats
import io

platform = cl.get_platforms()[1]
device = platform.get_devices()[0]
cntxt = cl.Context([device])
queue = cl.CommandQueue(cntxt)

def initialize(count):
    num_elem = count
    t = 10
    delt_t = 0.1
    Time = np.linspace(0, t, int(t / delt_t))

    pos_x = np.random.randint(0, 1000, count)
    pos_y = np.random.randint(0, 1000, count)
    v_x = np.random.randint(0, 100, count)
    v_y = np.random.randint(0, 100, count)

    a_x = np.random.randint(0, 100, count)
    a_y = np.random.randint(0, 100, count)
    a = []
    for i in range(count):
        a.append([a_x[i], a_y[i]])
    # print(a)

    G = 6.67 / 10**11

    mas = np.random.randint(0, 100, count)

    return num_elem, Time, delt_t, pos_x, pos_y, v_x, v_y, a, G, mas

def Verlet(num_elem, Time, delt_t, pos_x, pos_y, v_x, v_y, a, G, mas):
    start_time = time.time()

    # Sol_verl = []

    # pos_x_buf = []
    # pos_y_buf = []
    # speed_x_buf = []
    # speed_y_buf = []

    # pos_x_i = []
    # pos_y_i = []
    # speed_x_i = []
    # speed_y_i = []

    for t_i in range(len(Time)):
        # for i in range(num_elem):
        #     pos_x_buf.append(pos_x[i])
        #     pos_y_buf.append(pos_y[i])
        #     speed_x_buf.append(v_x[i])
        #     speed_y_buf.append(v_y[i])
        # pos_x_i.append(pos_x_buf)
        # pos_y_i.append(pos_y_buf)
        # speed_x_i.append(speed_x_buf)
        # speed_y_i.append(speed_y_buf)

        # pos_x_buf = []
        # pos_y_buf = []
        # speed_x_buf = []
        # speed_y_buf = []

        for i in range(num_elem):
            pos_x[i] = pos_x[i] + v_x[i] * delt_t + 0.5 * a[i][0] * delt_t**2
            pos_y[i] = pos_y[i] + v_y[i] * delt_t + 0.5 * a[i][1] * delt_t**2

        a_old = a
        for i in range(num_elem):
            a[i][0] = 0.0
            a[i][1] = 0.0
            for j in range(num_elem):
                if j != i and i != 0:
                    if (pos_x[j] - pos_x[i])**2 + (pos_y[j] - pos_y[i])**2 == 0.0:
                        a[i][0] += 0.0
                    else:
                        a[i][0] += G * mas[j] * (pos_x[j] - pos_x[i]) / \
                                                            (math.sqrt( (pos_x[j] - pos_x[i])**2 + (pos_y[j] - pos_y[i])**2 ))**3

                    if (pos_x[j] - pos_x[i])**2 + (pos_y[j] - pos_y[i])**2 == 0.0:
                        a[i][1] += 0.0
                    else:
                        a[i][1] += G * mas[j] * (pos_y[j] - pos_y[i]) / \
                                                            (math.sqrt( (pos_x[j] - pos_x[i])**2 + (pos_y[j] - pos_y[i])**2 ))**3
        
        for i in range(num_elem):
            v_x[i] = v_x[i] + 0.5 * (a[i][0] + a_old[i][0]) * delt_t
            v_y[i] = v_y[i] + 0.5 * (a[i][1] + a_old[i][1]) * delt_t
        
    # Sol_verl = np.concatenate((pos_x_i, pos_y_i, speed_x_i, speed_y_i), axis=-1)

    end_time = time.time()
    return end_time - start_time

#------------------------------------------------------------------

def V_th(v_x, v_y, a, a_old, delt_t, i):
    v_out_x = v_x[i] + 0.5 * (a[i][0] + a_old[i][0]) * delt_t
    v_out_y = v_y[i] + 0.5 * (a[i][1] + a_old[i][1]) * delt_t
    return v_out_x, v_out_y, i

def Verlet_thr(num_elem, Time, delt_t, pos_x, pos_y, v_x, v_y, a, G, mas):
    start_time = time.time()

    # Sol_verl_thr = []

    # pos_x_buf = []
    # pos_y_buf = []
    # speed_x_buf = []
    # speed_y_buf = []

    # pos_x_i = []
    # pos_y_i = []
    # speed_x_i = []
    # speed_y_i = []

    for t_i in range(len(Time)):
        # for i in range(num_elem):
        #     pos_x_buf.append(pos_x[i])
        #     pos_y_buf.append(pos_y[i])
        #     speed_x_buf.append(v_x[i])
        #     speed_y_buf.append(v_y[i])
        # pos_x_i.append(pos_x_buf)
        # pos_y_i.append(pos_y_buf)
        # speed_x_i.append(speed_x_buf)
        # speed_y_i.append(speed_y_buf)

        # pos_x_buf = []
        # pos_y_buf = []
        # speed_x_buf = []
        # speed_y_buf = []

        for i in range(num_elem):
            pos_x[i] = pos_x[i] + v_x[i] * delt_t + 0.5 * a[i][0] * delt_t**2
            pos_y[i] = pos_y[i] + v_y[i] * delt_t + 0.5 * a[i][1] * delt_t**2

        a_old = a
        with ThreadPoolExecutor(max_workers = num_elem) as executor:
            jobs = []
            for i in range(num_elem):
                jobs.append(executor.submit(A_th, pos_x = pos_x, pos_y = pos_y, G = G, mas = mas, pl_i = i, size = num_elem ))
            
            for job in as_completed(jobs):
                result_done = job.result()
                i = result_done[-1]
                a[i][0] = result_done[0]
                a[i][1] = result_done[1]

        for i in range(num_elem):
            v_x[i] = v_x[i] + 0.5 * (a[i][0] + a_old[i][0]) * delt_t
            v_y[i] = v_y[i] + 0.5 * (a[i][1] + a_old[i][1]) * delt_t

    # Sol_verl_thr = np.concatenate((pos_x_i, pos_y_i, speed_x_i, speed_y_i), axis=-1)

    end_time = time.time()
    return end_time - start_time

#------------------------------------------------------------------

def Pos_prosess(Time, num_elem, init_pos_x, init_pos_y, pos_queue, v_queue, G, mas, delt_t, return_pos_queue):
    pos_x = np.zeros((Time, num_elem))
    pos_y = np.zeros((Time, num_elem))
    pos_x[0], pos_y[0] = init_pos_x, init_pos_y
    for n in range(0, Time - 1):
        a_x, a_y = np.zeros(num_elem), np.zeros(num_elem)
        for i in range(num_elem):
            a_x[i], a_y[i], k = A_th(pos_x[n], pos_y[n], G, mas, i, num_elem)
        v_x, v_y = v_queue.get()
        for i in range(num_elem):
            pos_x[n + 1, i] = pos_x[n, i] + v_x[i] * delt_t + 0.5 * a_x[i] * delt_t ** 2
            pos_y[n + 1, i] = pos_y[n, i] + v_y[i] * delt_t + 0.5 * a_y[i] * delt_t ** 2
        pos_queue.put([pos_x[n + 1], pos_y[n + 1], a_x, a_y])
    return_pos_queue.put([pos_x, pos_y])

def Speed_prosess(Time, num_elem, init_v_x, init_v_y, pos_queue, speed_queue, G, mas, delt_t, return_speed_queue):
    v_x = np.zeros((Time, num_elem))
    v_y = np.zeros((Time, num_elem))
    v_x[0], v_y[0] = init_v_x, init_v_y
    speed_queue.put([v_x[0], v_y[0]])
    for n in range(0, Time - 1):
        pos_x, pos_y, old_a_x, old_a_y = pos_queue.get()
        for i in range(num_elem):
            a_x, a_y, k = A_th(pos_x, pos_y, G, mas, i, num_elem)
            v_x[n + 1, i] = v_x[n, i]  + 0.5 * (a_x + old_a_x[i]) * delt_t
            v_y[n + 1, i] = v_y[n, i]  + 0.5 * (a_y + old_a_y[i]) * delt_t
        speed_queue.put([v_x[n + 1], v_y[n + 1]])
    return_speed_queue.put([v_x, v_y])


def Verlet_proc(num_elem, Time, delt_t, pos_x, pos_y, v_x, v_y, a, G, mas):
    start_time = time.time()

    # Sol_verl_proc = []

    pos_queue, speed_queue = Queue(), Queue()
    return_pos_queue, return_speed_queue = Queue(), Queue()
    
    pos_prosess = Process(target = Pos_prosess, args=(len(Time), num_elem, pos_x, pos_y, pos_queue, speed_queue, G, mas, delt_t, return_pos_queue))
    speed_prosess = Process(target = Speed_prosess, args=(len(Time), num_elem, v_x, v_y, pos_queue, speed_queue, G, mas, delt_t, return_speed_queue))

    pos_prosess.start()
    speed_prosess.start()

    x_pos, y_pos = return_pos_queue.get()
    pos_prosess.join()

    v_x, v_y = return_speed_queue.get()
    speed_prosess.join()

    # Sol_verl_proc = np.concatenate((x_pos, y_pos, v_x, v_y), axis=-1)

    end_time = time.time()
    return end_time - start_time

def Verlet_proc_pool(num_elem, Time, delt_t, pos_x, pos_y, v_x, v_y, a, G, mas):
    start_time = time.time()

    # Sol_verl_proc = []

    # pos_x_buf = []
    # pos_y_buf = []
    # speed_x_buf = []
    # speed_y_buf = []

    # pos_x_i = []
    # pos_y_i = []
    # speed_x_i = []
    # speed_y_i = []

    with ProcessPoolExecutor(max_workers = 4) as executor:
        for t_i in range(len(Time)):
            # for i in range(num_elem):
            #     pos_x_buf.append(pos_x[i])
            #     pos_y_buf.append(pos_y[i])
            #     speed_x_buf.append(v_x[i])
            #     speed_y_buf.append(v_y[i])
            # pos_x_i.append(pos_x_buf)
            # pos_y_i.append(pos_y_buf)
            # speed_x_i.append(speed_x_buf)
            # speed_y_i.append(speed_y_buf)

            # pos_x_buf = []
            # pos_y_buf = []
            # speed_x_buf = []
            # speed_y_buf = []

            # jobs = []
            # for i in range(num_elem):
            #     jobs.append(executor.submit(Pos_proc, pos_x = pos_x, pos_y = pos_y, v_x = v_x, v_y = v_y, a = a, delt_t = delt_t, i = i ))
            
            # for job in as_completed(jobs):
            #     result_done = job.result()
            #     i = result_done[-1]
            #     pos_x[i] = result_done[0]
            #     pos_y[i] = result_done[1]

            for i in range(num_elem):
                pos_x[i] = pos_x[i] + v_x[i] * delt_t + 0.5 * a[i][0] * delt_t**2
                pos_y[i] = pos_y[i] + v_y[i] * delt_t + 0.5 * a[i][1] * delt_t**2

            a_old = a
            jobs = []
            for i in range(num_elem):
                jobs.append(executor.submit(A_th, pos_x = pos_x, pos_y = pos_y, G = G, mas = mas, pl_i = i, size = num_elem ))
            
            for job in as_completed(jobs):
                result_done = job.result()
                i = result_done[-1]
                a[i][0] = result_done[0]
                a[i][1] = result_done[1]

            # jobs = []
            # for i in range(num_elem):
            #     jobs.append(executor.submit(Speed_proc, v_x = v_x, v_y = v_y, a = a, a_old = a_old, delt_t = delt_t, i = i ))
            
            # for job in as_completed(jobs):
            #     result_done = job.result()
            #     i = result_done[-1]
            #     v_x[i] = result_done[0]
            #     v_y[i] = result_done[1]

            for i in range(num_elem):
                v_x[i] = v_x[i] + 0.5 * (a[i][0] + a_old[i][0]) * delt_t
                v_y[i] = v_y[i] + 0.5 * (a[i][1] + a_old[i][1]) * delt_t


    # Sol_verl_proc = np.concatenate((pos_x_i, pos_y_i, speed_x_i, speed_y_i), axis=-1)

    end_time = time.time()
    return end_time - start_time

def Pos_proc(pos_x, pos_y, v_x, v_y, a, delt_t, i):
    out_pos_x = pos_x[i] + v_x[i] * delt_t + 0.5 * a[i][0] * delt_t**2
    out_pos_y = pos_y[i] + v_y[i] * delt_t + 0.5 * a[i][1] * delt_t**2

    return out_pos_x, out_pos_y, i

def Speed_proc(v_x, v_y, a, a_old, delt_t, i):
    out_v_x = v_x[i] + 0.5 * (a[i][0] + a_old[i][0]) * delt_t
    out_v_y = v_y[i] + 0.5 * (a[i][1] + a_old[i][1]) * delt_t

    return out_v_x, out_v_y, i
#------------------------------------------------------------------

def Verlet_Cyth(num_elem, Time, delt_t, pos_x, pos_y, v_x, v_y, a, G, mas):
    start_time = time.time()

    # Sol_verl_cyth = []

    # pos_x_buf = []
    # pos_y_buf = []
    # speed_x_buf = []
    # speed_y_buf = []

    # pos_x_i = []
    # pos_y_i = []
    # speed_x_i = []
    # speed_y_i = []

    for t_i in range(len(Time)):
        # for i in range(num_elem):
        #     pos_x_buf.append(pos_x[i])
        #     pos_y_buf.append(pos_y[i])
        #     speed_x_buf.append(v_x[i])
        #     speed_y_buf.append(v_y[i])
        # pos_x_i.append(pos_x_buf)
        # pos_y_i.append(pos_y_buf)
        # speed_x_i.append(speed_x_buf)
        # speed_y_i.append(speed_y_buf)

        # pos_x_buf = []
        # pos_y_buf = []
        # speed_x_buf = []
        # speed_y_buf = []

        for i in range(num_elem):
            pos_x[i] = pos_x[i] + v_x[i] * delt_t + 0.5 * a[i][0] * delt_t**2
            pos_y[i] = pos_y[i] + v_y[i] * delt_t + 0.5 * a[i][1] * delt_t**2

        a_old = a
        for i in range(num_elem):
            res = A_cyth.A_th(list(pos_x), list(pos_y), G, list(mas), i, num_elem)

            a[i][0] = res[0]
            a[i][1] = res[1]

        for i in range(num_elem):
            v_x[i] = v_x[i] + 0.5 * (a[i][0] + a_old[i][0]) * delt_t
            v_y[i] = v_y[i] + 0.5 * (a[i][1] + a_old[i][1]) * delt_t

    # Sol_verl_cyth = np.concatenate((pos_x_i, pos_y_i, speed_x_i, speed_y_i), axis=-1)

    end_time = time.time()
    return end_time - start_time

#------------------------------------------------------------------

def Verlet_CL(num_elem, Time, delt_t, pos_x, pos_y, v_x, v_y, a, G, mas):
    start_time = time.time()
    
    # Sol_verl_cl = []

    # pos_x_buf0 = []
    # pos_y_buf0 = []
    # speed_x_buf0 = []
    # speed_y_buf0 = []

    # pos_x_i = []
    # pos_y_i = []
    # speed_x_i = []
    # speed_y_i = []

    code = """
    __kernel void A_cl(__global float* pos_x, __global float* pos_y, __global float* mas, __global float* res, float G, int pl_i, int size)  
    {
        res[0] = 0.0;
        res[1] = 0.0;
        for (int j = 0; j < size; j++)
        {
            if (j != pl_i && pl_i != 0)
            {
                if ((pos_x[j] - pos_x[pl_i])*(pos_x[j] - pos_x[pl_i]) + (pos_y[j] - pos_y[pl_i])*(pos_y[j] - pos_y[pl_i]) == 0.0)
                {
                    res[0] += 0.0;
                }
                else
                {
                    res[0] += G * mas[j] * (pos_x[j] - pos_x[pl_i]) / 
                        ( (sqrt( (pos_x[j] - pos_x[pl_i])*(pos_x[j] - pos_x[pl_i]) + (pos_y[j] - pos_y[pl_i])*(pos_y[j] - pos_y[pl_i]) ))*
                        (sqrt( (pos_x[j] - pos_x[pl_i])*(pos_x[j] - pos_x[pl_i]) + (pos_y[j] - pos_y[pl_i])*(pos_y[j] - pos_y[pl_i]) ))*
                        (sqrt( (pos_x[j] - pos_x[pl_i])*(pos_x[j] - pos_x[pl_i]) + (pos_y[j] - pos_y[pl_i])*(pos_y[j] - pos_y[pl_i]) )) );
                }
                
                if ((pos_x[j] - pos_x[pl_i])*(pos_x[j] - pos_x[pl_i]) + (pos_y[j] - pos_y[pl_i])*(pos_y[j] - pos_y[pl_i]) == 0.0)
                {
                    res[1] += 0.0;
                }
                else
                {
                    res[1] += G * mas[j] * (pos_y[j] - pos_y[pl_i]) / 
                        ( (sqrt( (pos_x[j] - pos_x[pl_i])*(pos_x[j] - pos_x[pl_i]) + (pos_y[j] - pos_y[pl_i])*(pos_y[j] - pos_y[pl_i]) ))*
                        (sqrt( (pos_x[j] - pos_x[pl_i])*(pos_x[j] - pos_x[pl_i]) + (pos_y[j] - pos_y[pl_i])*(pos_y[j] - pos_y[pl_i]) ))*
                        (sqrt( (pos_x[j] - pos_x[pl_i])*(pos_x[j] - pos_x[pl_i]) + (pos_y[j] - pos_y[pl_i])*(pos_y[j] - pos_y[pl_i]) )) );
                }
            }
        }
    }
    """

    bld = cl.Program(cntxt, code).build()

    for t_i in range(len(Time)):
        # for i in range(num_elem):
        #     pos_x_buf0.append(pos_x[i])
        #     pos_y_buf0.append(pos_y[i])
        #     speed_x_buf0.append(v_x[i])
        #     speed_y_buf0.append(v_y[i])
        # pos_x_i.append(pos_x_buf0)
        # pos_y_i.append(pos_y_buf0)
        # speed_x_i.append(speed_x_buf0)
        # speed_y_i.append(speed_y_buf0)

        # pos_x_buf0 = []
        # pos_y_buf0 = []
        # speed_x_buf0 = []
        # speed_y_buf0 = []

        for i in range(num_elem):
            pos_x[i] = pos_x[i] + v_x[i] * delt_t + 0.5 * a[i][0] * delt_t**2
            pos_y[i] = pos_y[i] + v_y[i] * delt_t + 0.5 * a[i][1] * delt_t**2

        a_old = a
        for i in range(num_elem):
            pos_x_buf = cl.Buffer(cntxt, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = np.array(pos_x).astype(np.float32))
            pos_y_buf = cl.Buffer(cntxt, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = np.array(pos_y).astype(np.float32))
            mas_buf = cl.Buffer(cntxt, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = np.array(mas).astype(np.float32))

            res = np.empty(2).astype(np.float32)
            res_buf = cl.Buffer(cntxt, cl.mem_flags.WRITE_ONLY, res.nbytes)
            
            launch = bld.A_cl(queue, [num_elem], None, pos_x_buf, pos_y_buf, mas_buf, res_buf, np.float32(G), np.int32(i), np.int32(num_elem))
            launch.wait()
            
            cl.enqueue_copy(queue, res, res_buf)
            
            a[i][0] = float(res[0])
            a[i][1] = float(res[1])

        for i in range(num_elem):
            v_x[i] = v_x[i] + 0.5 * (a[i][0] + a_old[i][0]) * delt_t
            v_y[i] = v_y[i] + 0.5 * (a[i][1] + a_old[i][1]) * delt_t

    # Sol_verl_cl = np.concatenate((pos_x_i, pos_y_i, speed_x_i, speed_y_i), axis=-1)

    end_time = time.time()
    return end_time - start_time

#------------------------------------------------------------------

def Plot(count, time_Verl, time_Verl_thr, time_Verl_proc, time_Verl_Cyth, time_Verl_CL):

    plt.subplot(1, 2, 1)
    plt.plot(count, time_Verl, 'b', label='Verlet')
    # plt.plot(count, time_Verl_thr, 'g', label='Verlet with threading')
    plt.plot(count, time_Verl_proc, 'r', label='Verlet with multiprocessing')
    plt.plot(count, time_Verl_Cyth, 'c', label='Verlet with Cython')
    plt.plot(count, time_Verl_CL, 'm', label='Verlet with OpenCL')
    plt.legend(loc='best')
    plt.xlabel('N')
    plt.grid()

    plt.subplot(1, 2, 2)
    # plt.plot(count, time_Verl / time_Verl_thr, 'g', label='Verlet with threading')
    plt.plot(count, time_Verl / time_Verl_proc, 'r', label='Verlet with multiprocessing')
    plt.plot(count, time_Verl / time_Verl_Cyth, 'c', label='Verlet with Cython')
    plt.plot(count, time_Verl / time_Verl_CL, 'm', label='Verlet with OpenCL')
    plt.legend(loc='best')
    plt.xlabel('N')
    plt.grid()

    plt.show()

#------------------------------------------------------------------
def A_th(pos_x, pos_y, G, mas, pl_i, size):
    a_x = 0
    a_y = 0
    for j in range(size):
        if j != pl_i and pl_i != 0:
            if (pos_x[j] - pos_x[pl_i])**2 + (pos_y[j] - pos_y[pl_i])**2 == 0.0:
                a_x += 0.0
            else:
                a_x += G * mas[j] * (pos_x[j] - pos_x[pl_i]) / \
                                    (np.sqrt( (pos_x[j] - pos_x[pl_i])**2 + (pos_y[j] - pos_y[pl_i])**2 ))**3
            
            if (pos_x[j] - pos_x[pl_i])**2 + (pos_y[j] - pos_y[pl_i])**2 == 0.0:
                a_y += 0.0
            else:
                a_y += G * mas[j] * (pos_y[j] - pos_y[pl_i]) / \
                                    (np.sqrt( (pos_x[j] - pos_x[pl_i])**2 + (pos_y[j] - pos_y[pl_i])**2 ))**3
    return a_x, a_y, pl_i

def main():
    # count = [20, 100, 200]
    count = [100]
    
    time_Verl = np.zeros(len(count))
    time_Verl_thr = np.zeros(len(count))
    time_Verl_proc = np.zeros(len(count))
    time_Verl_Cyth = np.zeros(len(count))
    time_Verl_CL = np.zeros(len(count))

    for c_i in range(len(count)):
        num_elem, Time, delt_t, pos_x, pos_y, v_x, v_y, a, G, mas = initialize(count[c_i])

        for i in range(1):
            time_Verl[c_i] += Verlet(num_elem, Time, delt_t, pos_x, pos_y, v_x, v_y, a, G, mas)
        time_Verl[c_i] /= 1
        print("Verlet ", " c_i= ", c_i, " time: ", time_Verl[c_i])

        # for i in range(3):
        #     time_Verl_thr[c_i] += Verlet_thr(num_elem, Time, delt_t, pos_x, pos_y, v_x, v_y, a, G, mas)
        # time_Verl_thr[c_i] /= 3
        # print("Verlet with threading ", " c_i= ", c_i, " time: ", time_Verl_thr[c_i])

        for i in range(1):
            time_Verl_proc[c_i] += Verlet_proc(num_elem, Time, delt_t, pos_x, pos_y, v_x, v_y, a, G, mas)
            #time_Verl_proc[c_i] += Verlet_proc_pool(num_elem, Time, delt_t, pos_x, pos_y, v_x, v_y, a, G, mas)
        time_Verl_proc[c_i] /= 1
        print("Verlet with multiprocessing ", " c_i= ", c_i, " time: ", time_Verl_proc[c_i])

        # for i in range(3):
        #     time_Verl_Cyth[c_i] += Verlet_Cyth(num_elem, Time, delt_t, pos_x, pos_y, v_x, v_y, a, G, mas)
        # time_Verl_Cyth[c_i] /= 3
        # print("Verlet with Cython ", " c_i= ", c_i, " time: ", time_Verl_Cyth[c_i])

        # for i in range(3):
        #     time_Verl_CL[c_i] += Verlet_CL(num_elem, Time, delt_t, pos_x, pos_y, v_x, v_y, a, G, mas)
        # time_Verl_CL[c_i] /= 3
        # print("Verlet with OpenCL ", " c_i= ", c_i, " time: ", time_Verl_CL[c_i])

        print()

    Plot(count, time_Verl, time_Verl_thr, time_Verl_proc, time_Verl_Cyth, time_Verl_CL)

if __name__ == '__main__':
    # pr = cProfile.Profile()
    # pr.enable()

    # main()

    # pr.disable()
    # s = io.StringIO()
    # ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    # ps.print_stats()

    # with open('E:/Python/Задание2/test.txt', 'w+') as f:
    #     f.write(s.getvalue())
    main()
