from libc.math cimport sqrt

def A_th(list pos_x, list pos_y, float G, list mas, int pl_i):
    cdef float a_x = 0
    cdef float a_y = 0
    cdef int j
    cdef list out
    for j in range(len(pos_x)):
        if j != pl_i and pl_i != 0:
            a_x += G * mas[j] * (pos_x[j] - pos_x[pl_i]) / \
                                    (sqrt( (pos_x[j] - pos_x[pl_i])**2 + (pos_y[j] - pos_y[pl_i])**2 ))**3
            
            a_y += G * mas[j] * (pos_y[j] - pos_y[pl_i]) / \
                                    (sqrt( (pos_x[j] - pos_x[pl_i])**2 + (pos_y[j] - pos_y[pl_i])**2 ))**3
    return [a_x, a_y, pl_i]