import matplotlib.pyplot as plt
import numpy as np

def surface_numerical(r, z, u):
    R, Z = np.meshgrid(r, z)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(R, Z, u)

    ax.set_xlabel('Ось r')
    ax.set_ylabel('Ось z')
    ax.set_zlabel('Ось u')

    plt.show()

def main():
    with open('output/u.txt', 'r') as f:
        u = np.array([list(map(float, s.split())) for s in f])
    with open('output/r.txt', 'r') as f:
        r = np.array(list(map(float, f.read().split())))
    with open('output/z.txt', 'r') as f:
        z = np.array(list(map(float, f.read().split())))

    surface_numerical(r, z, u)

if __name__ == "__main__":
    main()