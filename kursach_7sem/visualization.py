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

def surface_error(u, u_a, r, z, u0):
    R, Z = np.meshgrid(r, z)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    u_er = (u - u_a) / u0
    ax.plot_surface(R, Z, u_er)

    ax.set_xlabel('Ось r')
    ax.set_ylabel('Ось z')
    ax.set_zlabel('Ось u')

    plt.show()
    return u_er

def srez_v_r0(z, u, u0):
    plt.plot(z, u[:, 0])
    plt.show()

def sigma(r, z, u, u_a, R):
    mid = int((len(z) - 1) /2)
    h_z = z[1] - z[0]

    k = 0
    while r[k + 1] <= R:
        k += 1
    
    s_r = -(u[mid + 1, :k - 1] - u[mid, :k - 1]) / h_z
    s_a = -(u_a[mid + 1, :k - 1] - u_a[mid, :k - 1]) / h_z

    
    plt.plot(r[:k - 1], s_a, label='analitic')
    plt.plot(r[:k - 1], s_r, label='numerical')
    plt.legend()
    plt.show()

    plt.plot(r[:k - 1], (s_r - s_a) / np.max(np.abs(s_a)), label='relative error')
    plt.legend()
    plt.show()

def main():
    with open('output/u_o.txt', 'r') as f:
        u = np.array([list(map(float, s.split())) for s in f])
    with open('output/r_o.txt', 'r') as f:
        r = np.array(list(map(float, f.read().split())))
    with open('output/z_o.txt', 'r') as f:
        z = np.array(list(map(float, f.read().split())))
    with open('output/R_o.txt', 'r') as f:
        R = float(f.read())
    with open('output/u0_o.txt', 'r') as f:
        u_0 = float(f.read())
    
    surface_numerical(r, z, u)
    u_a = surface_analitic(r, z, R, u_0)
    u_er = surface_error(u, u_a, r, z, u_0)
    print(f"Макс значение относ погрешности (R={R:.0f}) - {np.max(np.abs(u_er)):.4f}")
    
    # различие погрешностей между разными порядками аппроксимации
    # surface_error(u1, u, r, z)
    srez_v_r0(z, u, u_0)
    sigma(r, z, u, u_a, R)


if __name__ == "__main__":
    main()