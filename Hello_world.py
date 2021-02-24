import numpy as np
import time
import matplotlib.pyplot as plt

def Mas_rand(size):
    det = 0
    while det==0:
        Mas = np.random.random((size, size))
        det = np.linalg.det(Mas)

    return Mas

def Sol_gauss(size):
    start_time = time.time()
    
    A = Mas_rand(size)
    f = np.random.sample(size)

    # sol = np.linalg.solve(A, f)
    
    for k in range(size):
        for i in range(k + 1, size):            
            div = A[i][k] / A[k][k]
            f[i] -= div * f[k]
            for j in range(k, size):
                A[i][j] -= div * A[k][j]

    sol = [0 for i in range(size)]
    for k in range(size - 1, -1, -1):
        sol[k] = (f[k] - sum([A[k][j] * sol[j] for j in range(k + 1, size)])) / A[k][k]

    end_time = time.time()
    t = end_time - start_time
    print("%f seconds\n" % t)
    # print(sol,'\n')

    return t
    

N = 20
t = np.zeros(N)
for n in range(N):
    t[n] = Sol_gauss(n)

plt.figure(figsize=(12, 7))
plt.plot(range(N), t, linewidth=2.0)
plt.show()
