import numpy as np

def Mas_rand(size):
    det = 0
    while det==0:
        Mas = np.random.sample((size, size))
        det = np.linalg.det(Mas)

    return Mas

def max_row(mas, k, size):
    max_elem = mas[k][k]
    max_ind = k

    for i in range(k + 1, size):
        if abs(mas[i][k]) > abs(max_elem):
            max_elem = mas[i][k]
            max_ind = i
    
    if max_ind != k:
        mas[k], mas[max_ind] = mas[max_ind], mas[k]

def Sol_gauss(size):
    A = Mas_rand(size)

    f = np.random.sample(size)

    # sol = np.linalg.solve(A, f)
    
    for k in range(size - 1):
        max_row(A, k, size)
        

    # return sol

Sol_gauss(5)