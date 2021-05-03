import numpy as np
import math

import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import Process, Queue

import sys
import time

import A_cyth

import pyopencl as cl

# platform = cl.get_platforms()[1]
# device = platform.get_devices()[0]
# cntxt = cl.Context([device])
# queue = cl.CommandQueue(cntxt)

G = 6.67 / 10**11
delt_t = 0.1

def initialize(count):
    num_elem = count
    t = 10
    Time = int(t / delt_t)

    # pos_x = np.random.randint(0, 1000, count)
    # pos_y = np.random.randint(0, 1000, count)
    # v_x = np.random.randint(0, 100, count)
    # v_y = np.random.randint(0, 100, count)

    # a_x = np.random.randint(0, 100, count)
    # a_y = np.random.randint(0, 100, count)

    pos_x = np.random.random(count)
    pos_y = np.random.random(count)
    v_x = np.random.random(count)
    v_y = np.random.random(count)
    
    a_x = np.random.random(count)
    a_y = np.random.random(count)
    a = []
    for i in range(count):
        a.append([a_x[i], a_y[i]])

    mas = np.random.random(count)

    return num_elem, Time, pos_x, pos_y, v_x, v_y, a, mas

def Verlet(num_elem, Time, pos_x, pos_y, v_x, v_y, a, mas):
    start_time = time.time()

    # pos_x_i = []
    # pos_y_i = []
    # speed_x_i = []
    # speed_y_i = []

    for t_i in range(Time):
        # pos_x_i.append(pos_x)
        # pos_y_i.append(pos_y)
        # speed_x_i.append(v_x)
        # speed_y_i.append(v_x)

        for i in range(num_elem):
            a[i][0], a[i][1], k = A_th(pos_x, pos_y, mas, i, num_elem)

        for i in range(num_elem):
            pos_x[i] = pos_x[i] + v_x[i] * delt_t + 0.5 * a[i][0] * delt_t**2
            pos_y[i] = pos_y[i] + v_y[i] * delt_t + 0.5 * a[i][1] * delt_t**2

        a_old = a
        for i in range(num_elem):
            a[i][0], a[i][1], k = A_th(pos_x, pos_y, mas, i, num_elem)
        
        for i in range(num_elem):
            v_x[i] = v_x[i] + 0.5 * (a[i][0] + a_old[i][0]) * delt_t
            v_y[i] = v_y[i] + 0.5 * (a[i][1] + a_old[i][1]) * delt_t
        
    # Sol_verl = np.concatenate((pos_x_i, pos_y_i, speed_x_i, speed_y_i), axis=-1)

    end_time = time.time()
    return end_time - start_time

#------------------------------------------------------------------

def Verlet_thr(num_elem, Time, pos_x, pos_y, v_x, v_y, a, mas):
    start_time = time.time()

    # pos_x_i = []
    # pos_y_i = []
    # speed_x_i = []
    # speed_y_i = []

    for t_i in range(Time):
        # pos_x_i.append(pos_x)
        # pos_y_i.append(pos_y)
        # speed_x_i.append(v_x)
        # speed_y_i.append(v_x)

        with ThreadPoolExecutor(max_workers = num_elem) as executor:
            jobs = []
            for i in range(num_elem):
                jobs.append(executor.submit(A_th, pos_x = pos_x, pos_y = pos_y, mas = mas, pl_i = i, size = num_elem ))
            
            for job in as_completed(jobs):
                result_done = job.result()
                i = result_done[-1]
                a[i][0] = result_done[0]
                a[i][1] = result_done[1]
        
        for i in range(num_elem):
            pos_x[i] = pos_x[i] + v_x[i] * delt_t + 0.5 * a[i][0] * delt_t**2
            pos_y[i] = pos_y[i] + v_y[i] * delt_t + 0.5 * a[i][1] * delt_t**2

        a_old = a
        with ThreadPoolExecutor(max_workers = num_elem) as executor:
            jobs = []
            for i in range(num_elem):
                jobs.append(executor.submit(A_th, pos_x = pos_x, pos_y = pos_y, mas = mas, pl_i = i, size = num_elem ))
            
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

def Pos_prosess(Time, num_elem, init_pos_x, init_pos_y, pos_queue, speed_queue, mas, return_pos_queue):
    pos_x = np.zeros((Time, num_elem))
    pos_y = np.zeros((Time, num_elem))

    pos_x[0], pos_y[0] = init_pos_x, init_pos_y

    for n in range(0, Time - 1):
        # v_x, v_y, a_x, a_y = speed_queue.get()

        a_x, a_y = np.zeros(num_elem), np.zeros(num_elem)
        for i in range(num_elem):
            a_x[i], a_y[i], k = A_th(pos_x[n], pos_y[n], mas, i, num_elem)
        v_x, v_y = speed_queue.get()
        
        for i in range(num_elem):
            pos_x[n + 1, i] = pos_x[n, i] + v_x[i] * delt_t + 0.5 * a_x[i] * delt_t ** 2
            pos_y[n + 1, i] = pos_y[n, i] + v_y[i] * delt_t + 0.5 * a_y[i] * delt_t ** 2
        pos_queue.put([pos_x[n + 1], pos_y[n + 1], a_x, a_y])
    return_pos_queue.put([pos_x, pos_y])

def Speed_prosess(Time, num_elem, init_v_x, init_v_y, init_a, pos_queue, speed_queue, mas, return_speed_queue):
    v_x = np.zeros((Time, num_elem))
    v_y = np.zeros((Time, num_elem))

    v_x[0], v_y[0] = init_v_x, init_v_y
    # a_x, a_y = init_a[:, 0], init_a[:, 1]

    # speed_queue.put([v_x[0], v_y[0], a_x, a_y])
    speed_queue.put([v_x[0], v_y[0]])
    for n in range(0, Time - 1):
        pos_x, pos_y, old_a_x, old_a_y = pos_queue.get()

        # a_x, a_y = np.zeros(num_elem), np.zeros(num_elem)
        # for i in range(num_elem):
        #     a_x[i], a_y[i], k = A_th(pos_x, pos_y, mas, i, num_elem)
        
        for i in range(num_elem):
            a_x, a_y, k = A_th(pos_x, pos_y, mas, i, num_elem)
            v_x[n + 1, i] = v_x[n, i]  + 0.5 * (a_x + old_a_x[i]) * delt_t
            v_y[n + 1, i] = v_y[n, i]  + 0.5 * (a_y + old_a_y[i]) * delt_t
            # v_x[n + 1, i] = v_x[n, i]  + 0.5 * (a_x[i] + old_a_x[i]) * delt_t
            # v_y[n + 1, i] = v_y[n, i]  + 0.5 * (a_y[i] + old_a_y[i]) * delt_t
        # speed_queue.put([v_x[n + 1], v_y[n + 1], a_x, a_y])
        speed_queue.put([v_x[n + 1], v_y[n + 1]])
    return_speed_queue.put([v_x, v_y])


def Verlet_proc(num_elem, Time, pos_x, pos_y, v_x, v_y, a, mas):
    start_time = time.time()

    a=np.array(a)

    pos_queue, speed_queue = Queue(), Queue()
    return_pos_queue, return_speed_queue = Queue(), Queue()
    
    pos_prosess = Process(target = Pos_prosess, args=(Time, num_elem, pos_x, pos_y, pos_queue, speed_queue, mas, return_pos_queue))
    speed_prosess = Process(target = Speed_prosess, args=(Time, num_elem, v_x, v_y, a, pos_queue, speed_queue, mas, return_speed_queue))

    pos_prosess.start()
    speed_prosess.start()

    x_pos, y_pos = return_pos_queue.get()
    pos_prosess.join()

    v_x, v_y = return_speed_queue.get()
    speed_prosess.join()

    # Sol_verl_proc = np.concatenate((x_pos, y_pos, v_x, v_y), axis=-1)

    end_time = time.time()
    return end_time - start_time

def Verlet_proc_pool(num_elem, Time, pos_x, pos_y, v_x, v_y, a, mas):
    start_time = time.time()

    # pos_x_i = []
    # pos_y_i = []
    # speed_x_i = []
    # speed_y_i = []

    for i in range(num_elem):
            a[i][0], a[i][1], k = A_th(pos_x, pos_y, mas, i, num_elem)
    
    with ProcessPoolExecutor(max_workers = 4) as executor:
        for t_i in range(Time):
            # pos_x_i.append(pos_x)
            # pos_y_i.append(pos_y)
            # speed_x_i.append(v_x)
            # speed_y_i.append(v_x)

            for i in range(num_elem):
                pos_x[i] = pos_x[i] + v_x[i] * delt_t + 0.5 * a[i][0] * delt_t**2
                pos_y[i] = pos_y[i] + v_y[i] * delt_t + 0.5 * a[i][1] * delt_t**2

            a_old = a
            jobs = []
            for i in range(num_elem):
                jobs.append(executor.submit(A_th, pos_x = pos_x, pos_y = pos_y, mas = mas, pl_i = i, size = num_elem ))
            
            for job in as_completed(jobs):
                result_done = job.result()
                i = result_done[-1]
                a[i][0] = result_done[0]
                a[i][1] = result_done[1]

            for i in range(num_elem):
                v_x[i] = v_x[i] + 0.5 * (a[i][0] + a_old[i][0]) * delt_t
                v_y[i] = v_y[i] + 0.5 * (a[i][1] + a_old[i][1]) * delt_t

    # Sol_verl_proc = np.concatenate((pos_x_i, pos_y_i, speed_x_i, speed_y_i), axis=-1)

    end_time = time.time()
    return end_time - start_time
#------------------------------------------------------------------

def Verlet_Cyth(num_elem, Time, pos_x, pos_y, v_x, v_y, a, mas):
    start_time = time.time()

    # pos_x_i = []
    # pos_y_i = []
    # speed_x_i = []
    # speed_y_i = []

    for t_i in range(Time):
        # pos_x_i.append(pos_x)
        # pos_y_i.append(pos_y)
        # speed_x_i.append(v_x)
        # speed_y_i.append(v_x)

        for i in range(num_elem):
            res = A_cyth.A_th(list(pos_x), list(pos_y), G, list(mas), i, num_elem)

            a[i][0] = res[0]
            a[i][1] = res[1]

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

def Verlet_CL(num_elem, Time, pos_x, pos_y, v_x, v_y, a, mas, N1, N2):
    platform = cl.get_platforms()[N1]
    device = platform.get_devices()[N2]
    cntxt = cl.Context([device])
    queue = cl.CommandQueue(cntxt)

    start_time = time.time()

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
                if (pos_x[j] != pos_x[pl_i])
                {
                    res[0] += G * mas[j] * (pos_x[j] - pos_x[pl_i]) / 
                        ( (sqrt( (pos_x[j] - pos_x[pl_i])*(pos_x[j] - pos_x[pl_i]) + (pos_y[j] - pos_y[pl_i])*(pos_y[j] - pos_y[pl_i]) ))*
                        (sqrt( (pos_x[j] - pos_x[pl_i])*(pos_x[j] - pos_x[pl_i]) + (pos_y[j] - pos_y[pl_i])*(pos_y[j] - pos_y[pl_i]) ))*
                        (sqrt( (pos_x[j] - pos_x[pl_i])*(pos_x[j] - pos_x[pl_i]) + (pos_y[j] - pos_y[pl_i])*(pos_y[j] - pos_y[pl_i]) )) );
                }
                
                if (pos_y[j] != pos_y[pl_i])
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

    for t_i in range(Time):
        # pos_x_i.append(pos_x)
        # pos_y_i.append(pos_y)
        # speed_x_i.append(v_x)
        # speed_y_i.append(v_x)

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

def Plot(count, time_Verl, time_Verl_thr, time_Verl_proc, time_Verl_proc_pool, time_Verl_Cyth, time_Verl_CL, time_Verl_CL00, time_Verl_CL10, time_Verl_CL11):

    # plt.subplot(1, 2, 1)
    # plt.plot(count, time_Verl, 'b', label='Verlet')
    # plt.plot(count, time_Verl_thr, 'g', label='Verlet with threading')
    # plt.plot(count, time_Verl_proc, 'y', label='Verlet with multiprocessing')
    # plt.plot(count, time_Verl_proc_pool, 'r', label='Verlet with multiprocessing pool')
    # plt.plot(count, time_Verl_Cyth, 'c', label='Verlet with Cython')
    # plt.plot(count, time_Verl_CL, 'm', label='Verlet with OpenCL')

    plt.plot(count, time_Verl_CL00, 'r', label='Verlet with OpenCL NVIDIA CUDA')
    plt.plot(count, time_Verl_CL10, 'g', label='Verlet with OpenCL Intel(R) HD Graphics 4600')
    plt.plot(count, time_Verl_CL11, 'b', label='Verlet with OpenCL Intel(R) Core(TM) i7-4710HQ CPU @ 2.50GHz')
    plt.legend(loc='best')
    plt.xlabel('N')
    plt.grid()

    # plt.subplot(1, 2, 2)
    # plt.plot(count, time_Verl / time_Verl_thr, 'g', label='Verlet with threading')
    # plt.plot(count, time_Verl / time_Verl_proc, 'y', label='Verlet with multiprocessing')
    # plt.plot(count, time_Verl / time_Verl_proc_pool, 'r', label='Verlet with multiprocessing pool')
    # plt.plot(count, time_Verl / time_Verl_Cyth, 'c', label='Verlet with Cython')
    # plt.plot(count, time_Verl / time_Verl_CL, 'm', label='Verlet with OpenCL')
    # plt.legend(loc='best')
    # plt.xlabel('N')
    # plt.grid()

    plt.show()

#------------------------------------------------------------------
def A_th(pos_x, pos_y, mas, pl_i, size):
    a_x = 0
    a_y = 0
    for j in range(size):
        if j != pl_i:
            if pos_x[j] != pos_x[pl_i]:
                a_x += G * mas[j] * (pos_x[j] - pos_x[pl_i]) / \
                                    (np.sqrt( (pos_x[j] - pos_x[pl_i])**2 + (pos_y[j] - pos_y[pl_i])**2 ))**3
            
            if pos_y[j] != pos_y[pl_i]:
                a_y += G * mas[j] * (pos_y[j] - pos_y[pl_i]) / \
                                    (np.sqrt( (pos_x[j] - pos_x[pl_i])**2 + (pos_y[j] - pos_y[pl_i])**2 ))**3
    return a_x, a_y, pl_i

def main():
    count = [100, 200, 400]
    # count = [20]
    repeat = 1
    
    time_Verl = np.zeros(len(count))
    time_Verl_thr = np.zeros(len(count))
    time_Verl_proc = np.zeros(len(count))
    time_Verl_proc_pool = np.zeros(len(count))
    time_Verl_Cyth = np.zeros(len(count))
    time_Verl_CL = np.zeros(len(count))

    time_Verl_CL00 = np.zeros(len(count)) # NVIDIA CUDA
    time_Verl_CL10 = np.zeros(len(count)) # Intel(R) HD Graphics 4600
    time_Verl_CL11 = np.zeros(len(count)) # Intel(R) Core(TM) i7-4710HQ CPU @ 2.50GHz

    for c_i in range(len(count)):
        num_elem, Time, pos_x, pos_y, v_x, v_y, a, mas = initialize(count[c_i])

        # for i in range(repeat):
        #     time_Verl_thr[c_i] += Verlet_thr(num_elem, Time, pos_x, pos_y, v_x, v_y, a, mas)
        # time_Verl_thr[c_i] /= repeat
        # print("Verlet with threading ", " c_i= ", c_i, " time: ", time_Verl_thr[c_i])

        # for i in range(repeat):
        #     time_Verl_proc[c_i] += Verlet_proc(num_elem, Time, pos_x, pos_y, v_x, v_y, a, mas)
        # time_Verl_proc[c_i] /= repeat
        # print("Verlet with multiprocessing ", " c_i= ", c_i, " time: ", time_Verl_proc[c_i])

        # for i in range(repeat):
        #     time_Verl_proc_pool[c_i] += Verlet_proc_pool(num_elem, Time, pos_x, pos_y, v_x, v_y, a, mas)
        # time_Verl_proc_pool[c_i] /= repeat
        # print("Verlet with multiprocessing pool ", " c_i= ", c_i, " time: ", time_Verl_proc_pool[c_i])

        # for i in range(repeat):
        #     time_Verl_Cyth[c_i] += Verlet_Cyth(num_elem, Time, pos_x, pos_y, v_x, v_y, a, mas)
        # time_Verl_Cyth[c_i] /= repeat
        # print("Verlet with Cython ", " c_i= ", c_i, " time: ", time_Verl_Cyth[c_i])

        # for i in range(repeat):
        #     time_Verl_CL[c_i] += Verlet_CL(num_elem, Time, pos_x, pos_y, v_x, v_y, a, mas, 1, 0)
        # time_Verl_CL[c_i] /= repeat
        # print("Verlet with OpenCL ", " c_i= ", c_i, " time: ", time_Verl_CL[c_i])

        # for i in range(repeat):
        #     time_Verl[c_i] += Verlet(num_elem, Time, pos_x, pos_y, v_x, v_y, a, mas)
        # time_Verl[c_i] /= repeat
        # print("Verlet ", " c_i= ", c_i, " time: ", time_Verl[c_i])

        for i in range(repeat):
            time_Verl_CL00[c_i] += Verlet_CL(num_elem, Time, pos_x, pos_y, v_x, v_y, a, mas, 0, 0)
        time_Verl_CL00[c_i] /= repeat
        print("Verlet with OpenCL NVIDIA CUDA ", " c_i= ", c_i, " time: ", time_Verl_CL00[c_i])

        for i in range(repeat):
            time_Verl_CL10[c_i] += Verlet_CL(num_elem, Time, pos_x, pos_y, v_x, v_y, a, mas, 1, 0)
        time_Verl_CL10[c_i] /= repeat
        print("Verlet with OpenCL Intel(R) HD Graphics 4600 ", " c_i= ", c_i, " time: ", time_Verl_CL10[c_i])

        for i in range(repeat):
            time_Verl_CL11[c_i] += Verlet_CL(num_elem, Time, pos_x, pos_y, v_x, v_y, a, mas, 1, 1)
        time_Verl_CL11[c_i] /= repeat
        print("Verlet with OpenCL Intel(R) Core(TM) i7-4710HQ CPU @ 2.50GHz ", " c_i= ", c_i, " time: ", time_Verl_CL11[c_i])

        print()

    Plot(count, time_Verl, time_Verl_thr, time_Verl_proc, time_Verl_proc_pool, time_Verl_Cyth, time_Verl_CL, time_Verl_CL00, time_Verl_CL10, time_Verl_CL11)

if __name__ == '__main__':
    main()