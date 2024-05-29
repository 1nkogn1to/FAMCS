import numpy as np
import matplotlib.pyplot as plt


def I(_r, _z):
    n = len(_r)
    h = _r[1] - _r[0]
    sum = _z[0] * _r[0] + _z[-1] * _r[-1]
    for i in range(1, n - 1):
        if i % 2 == 0:
            sum += 2 * _z[i] * _r[i]
        else:
            sum += 4 * _z[i] * _r[i]
    
    return -2 * np.pi * h / 3 * sum

def k(_ind, _r, _z):
    h = _r[1] - _r[0]
    return (_r[_ind] - h / 2) / np.sqrt(1 + ((_z[_ind] - _z[_ind - 1]) / h)**2)

def a(_ind, _r, _z):
    h = _r[1] - _r[0]
    return k(_ind, _r, _z) / h**2

def b(_ind, _r, _z, _Bo, _I):
    h = _r[1] - _r[0]
    return _Bo * _r[_ind] / _I**(2/3) - (k(_ind + 1, _r, _z) + k(_ind, _r, _z)) / h**2

def c(_ind, _r, _z):
    h = _r[1] - _r[0]
    return k(_ind + 1, _r, _z) / h**2

def condition():
    return False

def RSweepMethod(_r):
    Bo = 1
    alpha = np.pi / 4

    h = _r[1] - _r[0]
    n = len(_r)
    A = np.zeros((n, n))
    d = np.zeros(n)
    z = np.zeros(n)
    z[1: -1] = _r[1:-1] - 1 # начальное
    
    A[-2, -2] = -4 / (2 * h)
    A[-2, -3] = 1 / (2 * h)
    d[-2] = np.tan(alpha)
    
    while True:
        temp = z
        integ = I(_r, temp)

        d[1:-2] = _r[1:-2] * (- Bo * integ**(1/3) / np.pi + 2 * np.sin(alpha))
        A[1, 1] = b(1, _r, temp, Bo, integ) + 4 / 3 * a(1, _r, temp)
        A[1, 2] = c(1, _r, temp) - 1 / 3 * a(1, _r, temp)

        # заполнение оставшейся части матрицы
        for i in range(2, n - 2):
            A[i, i - 1] = a(i, _r, temp)
            A[i, i] = b(i, _r, temp, Bo, integ)
            A[i, i + 1] = c(i, _r, temp)

        break

    


def main():
    N = 10
    r = np.linspace(0, 1, N)
    RSweepMethod(r)

if __name__ == "__main__":
    main()