import numpy as np
import matplotlib.pyplot as plt


def read(filename):
    with open(filename, "r") as f:
        u = np.array([list(map(float, s.split())) for s in f])
    
    return u

def func(l):
    return l + np.tan(l)

def dihotomy(f, a, b, accuracy=1e-8):
    while b - a > accuracy:
        mid = (a + b) / 2
        if f(mid) < 0:
            a = mid
        elif f(mid) > 0:
            b = mid
        else:
            return mid
    return b

def countLs(how_many=40, eps=1e-4):
    return [dihotomy(func, np.pi * k / 2 + eps, np.pi * k) for k in range(1, how_many + 1)]

def norm(lam):
    return (2 + 2 * lam**2) / (2 + lam**2)

def fik(lam):
    return norm(lam) * (lam * np.sin(lam) + (lam**2 + 4) * np.cos(lam) - 4) / (2 * lam**3)

def fk(lam):
    return norm(lam) * (2 * np.cos(lam) - 2) / lam

def analitic(x):
    t = np.linspace(0, 1, len(x))
    u = np.zeros((len(t), len(x)))
    numOfLs = 40
    ls = countLs(numOfLs)

    for i in range(len(t)):
        for j in range(len(x)):
            u[i, j] += 3 / 2 * x[j] + t[i]**3
            for k in range(numOfLs):
                u[i, j] += ((fik(ls[k]) - fk(ls[k]) / ls[k]**2) * np.cos(ls[k] * t[i]) + fk(ls[k]) / ls[k]**2) * np.sin(ls[k] * x[j])

    return u

def plot(x, u, title=''):
    t = np.linspace(0, 1, len(u))
    tau = t[1] - t[0]
    X, T = np.meshgrid(x, t)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, T, u, cmap='viridis', label='u(x,t)')

    fig.colorbar(surf)
    if title == '':
        plt.title(f'Визуализация для шага $\\tau ={tau}$')
    else:
        plt.title(title)
    plt.xlabel('Ox')
    plt.ylabel('Ot')
    plt.show()

def plot_error(x, u, ua, tau):
    plt.title(f"Аналитическое решение и Численное с шагом $\\tau = {tau}$")
    plt.plot(x, u - ua)
    plt.show()

def main():
    u1 = read("output1.txt")
    u001 = read("output001.txt")
    x = np.linspace(0, 1, len(u1[0]))

    ua = analitic(x)
    plot(x, ua, "Аналитическое решение")
    plot(x, u1)
    plot(x, u001)
    plot_error(x, u1[-1], ua[-1], 1 / (len(u1) - 1))
    plot_error(x, u001[-1], ua[-1], 1 / (len(u001) - 1))

if __name__ == "__main__":
    main()