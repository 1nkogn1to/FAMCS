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

def u(r, z, R, u_0):
    N = len(r[0])
    M = len(z)
    u = np.array(np.zeros((M, N)))
    r_s_2 = r**2 + z**2
    
    for i in range(N):
        for j in range(M):
            val = r_s_2[j, i] - R**2 + ((r_s_2[j, i] - R**2)**2 + 4 * R**2 * z[j, 0]**2)**(1/2)
            if r[0, i] <= R and z[j, 0] == 0:
                u[j, i] = u_0
            else:
                u[j, i] = 2 * u_0 / np.pi * np.arctan(np.sqrt(2) * R * (val)**(-1/2))

    return u

def surface_analitic(r, z, b, u_0):
    R, Z = np.meshgrid(r, z)
    U = u(R, Z, b, u_0)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(R, Z, U)

    ax.set_xlabel('Ось r')
    ax.set_ylabel('Ось z')
    ax.set_zlabel('Ось u')

    plt.show()

    return U

def surface_error(u, u_a, r, z):
    R, Z = np.meshgrid(r, z)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    u_er = u - u_a
    ax.plot_surface(R, Z, u_er)

    ax.set_xlabel('Ось r')
    ax.set_ylabel('Ось z')
    ax.set_zlabel('Ось u')

    plt.show()
    return u_er

def srez_v_z0(r, z, u, u_a):
    m = len(z)
    plt.plot(r, np.abs(u[int((m - 1) / 2), :] - u_a[int((m - 1) / 2), :]))
    plt.show()

def srez_v_z_min(r, u, u_a):
    plt.plot(r, np.abs(u[0, :] - u_a[0, :]))
    plt.show()

def srez_v_r0(z, u, u_a):
    plt.plot(z, np.abs(u[:, 0] - u_a[:, 0]))
    plt.show()

def srez_v_r_max(z, u, u_a):
    plt.plot(z, np.abs(u[:, -1] - u_a[:, -1]))
    plt.show()

def sigma(r, z, u, u_a, R):
    mid = int((len(z) - 1) /2)
    h_z = z[1] - z[0]

    k = 0
    while r[k + 1] <= R:
        k += 1
    
    s_r = (u[mid + 1, :k - 1] - u[mid, :k - 1]) / h_z
    s_a = (u_a[mid + 1, :k - 1] - u_a[mid, :k - 1]) / h_z
    print(r[k], R, sep=' ')
    plt.plot(r[:k - 1], s_a, label='analitic')
    plt.plot(r[:k - 1], s_r, label='numerical')
    plt.legend()
    plt.show()

def main():
    #with open('u_o.txt', 'r') as f:
        #u = np.array([list(map(float, s.split())) for s in f])
    with open('r_o.txt', 'r') as f:
        r = np.array(list(map(float, f.read().split())))
    with open('z_o.txt', 'r') as f:
        z = np.array(list(map(float, f.read().split())))
    
    with open('R_o.txt', 'r') as f:
        R = float(f.read())
    with open('u0_o.txt', 'r') as f:
        u_0 = float(f.read())
    with open('u_o1.txt', 'r') as f:
        u1 = np.array([list(map(float, s.split())) for s in f])
    
    #surface_numerical(r, z, u1)
    u_a = surface_analitic(r, z, R, u_0)
    #u_er1 = surface_error(u, u_a, r, z)
    #surface_error(u1, u_a, r, z)
    # различие погрешностей между разными порядками аппроксимации
    # surface_error(u1, u, r, z)
    #srez_v_z0(r, z, u1, u_a)
    #srez_v_z_min(r, u1, u_a)
    #srez_v_r0(z, u1, u_a)
    #srez_v_r_max(z, u1, u_a)
    sigma(r, z, u1, u_a, R)


if __name__ == "__main__":
    main()