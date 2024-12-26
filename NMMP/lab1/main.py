import numpy as np
import matplotlib.pyplot as plt

kappa0 = 1
g0 = -2
kappa1 = 0
g1 = np.cos(1)**2

def f(x):
    return x * np.sin(2 * x)

def k(x):
    return np.cos(x)**2

def dk(x):
    return -np.sin(2 * x)

def q(x):
    return np.sin(2 * x)

def progonka(matr, d):
    n = len(d)
    alpha = np.zeros(n)
    beta = np.zeros(n)
    solution = np.zeros(n)
    for i in range(n - 1):
        alpha[i + 1] = -matr[i, 2] / (matr[i, 0] * alpha[i] + matr[i, 1])
        beta[i + 1] = (d[i] - matr[i, 0] * beta[i]) / (matr[i, 0] * alpha[i] + matr[i, 1])
    
    solution[n - 1] = (d[n - 1] - matr[n - 1, 0] * beta[n - 1]) / (matr[n - 1, 0] * alpha[n - 1] + matr[n - 1, 1])

    for i in range(n - 1, 0, -1):
        solution[i - 1] = alpha[i] * solution[i] + beta[i]

    return solution

def task1(x):
    n = len(x)
    syst = np.zeros((n, 3))
    d = np.zeros(n)
    h = x[1] - x[0]

    _kappa0 = kappa0 + h / 2 * q(x[0]) - h / 2 * dk(x[0]) / k(x[0]) * kappa0
    _kappa1 = kappa1 + h / 2 * q(x[-1]) + h / 2 * dk(x[-1]) / k(x[-1]) * kappa1

    d[0] = g0 - h / 2 * dk(x[0]) / k(x[0]) * g0 + h / 2 * f(x[0])
    d[-1] = g1 + h / 2 * dk(x[-1]) / k(x[-1]) * g1 + h / 2 * f(x[-1])

    syst[0, 1] = (k(x[0]) / h + _kappa0)
    syst[0, 2] = -k(x[0]) / h
    syst[-1, 0] = -k(x[-1]) / h
    syst[-1, 1] = (k(x[-1]) / h + _kappa1)

    for i in range(1, n - 1):
        syst[i, 0] = k(x[i]) / h**2 - dk(x[i]) / (2 * h)
        syst[i, 1] = -(2 * k(x[i]) / h**2 + q(x[i]))
        syst[i, 2] = k(x[i]) / h**2 + dk(x[i]) / (2 * h)
        d[i] = -f(x[i])

    return progonka(syst, d)

def task2(x):
    n = len(x)
    syst = np.zeros((n, 3))
    fi = np.zeros(n)
    d = np.zeros(n)
    h = x[1] - x[0]

    a = np.zeros(n)
    for i in range(1, n):
        a[i] = k(x[i] - h / 2)

    d[0] = q(x[0] + h / 4)
    d[-1] = q(x[-1] - h / 4)
    fi[0] = g0 + h / 2 * f(x[0] + h / 4)
    fi[-1] = g1 + h / 2 * f(x[-1] - h / 4)

    syst[0, 1] = kappa0 + h / 2 * d[0] + a[1] / h
    syst[0, 2] = -a[1] / h
    syst[-1, 0] = -a[-1] / h
    syst[-1, 1] = kappa1 + h / 2 * d[-1] + a[-1] / h

    for i in range(1, n - 1):
        d[i] = q(x[i])    
        syst[i, 0] = a[i] / h**2
        syst[i, 1] = -((a[i + 1] + a[i]) / h**2 + d[i])
        syst[i, 2] = a[i + 1] / h**2
        fi[i] = -f(x[i])

    return progonka(syst, fi)

def task3(x):
    n = len(x)
    syst = np.zeros((n, 3))
    d = np.zeros(n)
    h = x[1] - x[0]

    syst[0, 1] = 1 / h * (k(h / 2) + q(h / 2) * (h / 2)**2) + kappa0
    syst[-1, 1] = 1 / h * (k(1 - h / 2) + q(1 - h / 2) * (h / 2)**2) + kappa1
    d[0] = f(h / 2) * h / 2 + g0
    d[-1] = f(1 - h / 2) * h / 2 + g1
    
    for i in range(1, n - 1):
        syst[i, 1] = 1 / h * (k(x[i]) + (h / 2)**2 * (q(x[i] - h / 2) + q(x[i] + h / 2)))
        d[i] = h / 2 * (f(x[i] - h / 2) + f(x[i] + h / 2))

    for i in range(n - 1):
        syst[i, 2] = 1 / h * (-k(x[i] + h / 2) + q(x[i] + h / 2) * (h / 2)**2)
        syst[i + 1, 0] = syst[i, 2]
    
    return progonka(syst, d)

def plot(x, u1, u2, u3):
    plt.plot(x, u1, label='Первый метод', marker='D')
    plt.plot(x, u2, label='Метод баланса')
    plt.plot(x, u3, label='Метод Ритца', marker='s')
    plt.legend()
    plt.xlabel('Ox')
    plt.ylabel('u(x)')
    plt.show()

def plot_error(x, ue):
    h = x[1] - x[0]
    plt.plot(x, ue)
    plt.title(f'Абсолютная погрешность двух методов при h={h}')
    plt.xlabel('Ox')
    plt.ylabel('u(x)')
    plt.show()

def main():
    N = 10
    x = np.linspace(0, 1, N + 1)
    u1 = task1(x)
    u2 = task2(x)
    u3 = task3(x)
    plot(x, u1, u2, u3)
    plot_error(x, u1 - u2)


if __name__ == "__main__":
    main()