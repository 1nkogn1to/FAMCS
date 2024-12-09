import numpy as np
import matplotlib.pyplot as plt


def read(filename):
    with open(filename, "r") as f:
        u = np.array([list(map(float, s.split())) for s in f])
    
    return u

def plot(x, u):
    t = np.linspace(0, 1, len(u))
    tau = t[1] - t[0]
    X, T = np.meshgrid(x, t)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, T, u, cmap='viridis', label='u(x,t)')

    fig.colorbar(surf)
    plt.title(f'Визуализация для шага $\\tau ={tau}$')
    plt.xlabel('Ox')
    plt.ylabel('Ot')
    plt.show()

def plot_error(x, u1, u2):
    plt.title("Погрешность на последнем слое при $\\tau = 0.1$ и $\\tau = 0.01$")
    plt.plot(x, u1 - u2)
    plt.show()

def main():
    u1 = read("output1.txt")
    u01 = read("output01.txt")
    u2 = read("output2.txt")
    x = np.linspace(0, 1, len(u1[0]))
    plot(x, u1)
    plot(x, u01)
    plot(x, u2)
    plot_error(x, u01[-1], u1[-1])

if __name__ == "__main__":
    main()