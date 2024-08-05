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

def RSweepMethod(_r):
    Bo = 1
    alpha = np.pi / 6
    eps = 1e-4

    h = _r[1] - _r[0]
    n = len(_r)
    A = np.zeros((n, n))
    d = np.zeros(n)
    z = np.zeros(n)
    z = _r - 1 # начальное
    
    counter = 1

    while True:
        temp = np.array(z)
        integ = I(_r, temp)
        print(integ)

        d[0] = h**2 / 2 * (-Bo * integ**(1/3) / np.pi + 2 * np.sin(alpha))
        d[1:-2] = -_r[1:-2] * (Bo * integ**(1/3) / np.pi - 2 * np.sin(alpha))
        d[-2] = -h * np.tan(alpha) - h**2 / (2 * np.cos(alpha)**3) * (Bo * integ**(1/3) / np.pi - np.sin(alpha))

        A[0, 0] = h**2 * Bo / (2*integ**(2/3)) - 1
        A[0, 1] = 1
        #A[-2, -2] = 1

        # заполнение оставшейся части матрицы
        for i in range(1, n - 2):
            A[i, i - 1] = a(i, _r, temp)
            A[i, i] = b(i, _r, temp, Bo, integ)
            A[i, i + 1] = c(i, _r, temp)
        
        print(A)
    
        Alp = np.zeros(n) # коэффициенты альфа для прогонки
        Bet = np.zeros(n) # коэффициенты бета для прогонки

        Alp[1] = -A[0, 1] / A[0, 0]
        Bet[1] = d[0] / A[0, 0]

        # прогонка
        for i in range(1, n - 1):
            Alp[i + 1] = - A[i, i + 1] / (A[i, i] - A[i, i - 1] * Alp[i])
            Bet[i + 1] = (d[i] - A[i, i - 1] * Bet[i]) / (A[i, i] - A[i, i - 1] * Alp[i])
        
        temp[n - 2] = Bet[n - 1]

        for i in range(n - 3, -1, -1):
            temp[i] = Alp[i + 1] * temp[i + 1] + Bet[i + 1]

        if (max(np.abs(temp - z)) < eps):
            z = np.array(temp)
            break

        counter += 1
        z = np.array(temp)
    print(counter)

    return z

def plot(_r, _z):
    plt.plot(_r, _z)
    plt.show()

def main():
    N = 20
    r = np.linspace(0, 1, N)
    z = RSweepMethod(r)

    integ = I(r, z)
    r /= (integ**(1/3))
    z /= (integ**(1/3))

    plot(r, z)

if __name__ == "__main__":
    main()