import numpy as np
import matplotlib.pyplot as plt


def plot(x, y, z):
    X, Y = np.meshgrid(x, y)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, z)

    ax.set_xlabel('Ось x')
    ax.set_ylabel('Ось y')
    ax.set_zlabel('Ось z')

    plt.show()


def main():
    with open("x.txt", "r") as f:
        x = np.array(list(map(float, f.read().split())))
    with open("y.txt", "r") as f:
        y = np.array(list(map(float, f.read().split())))
    with open("z.txt", "r") as f:
        z = np.array([list(map(float, s.split())) for s in f])

    plot(x, y, z)
    

if __name__ == "__main__":
    main()