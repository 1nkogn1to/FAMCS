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

def sigma(r, z, u, R):
    mid = int((len(z) - 1) /2)
    h_z = z[1] - z[0]

    k = 0
    while r[k + 1] <= R:
        k += 1
    
    s_r = (u[mid + 1, :k - 1] - u[mid, :k - 1]) / h_z
    
    plt.plot(r[:k - 1], s_r, label='numerical')
    plt.legend()
    plt.show()

def main():
    with open('r_n.txt', 'r') as f:
        r = np.array(list(map(float, f.read().split())))
    with open('z_n.txt', 'r') as f:
        z = np.array(list(map(float, f.read().split())))
    
    with open('R_n.txt', 'r') as f:
        R = float(f.read())
    with open('u_n.txt', 'r') as f:
        u1 = np.array([list(map(float, s.split())) for s in f])
    
    surface_numerical(r, z, u1)
    sigma(r, z, u1, R)


if __name__ == "__main__":
    main()