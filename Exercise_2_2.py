import numpy as np
import matplotlib.pyplot as plt


EA = 1
n0 = 1
L = 1

def element_stiffness(EA,Le):
    return (EA / Le) * np.array([[1, -1],[-1, 1]])

def element_force(n0, Le):
    return (n0 * Le/2) * np.array([1, 1])

def fem_solver(ne):
    nn = ne + 1
    Le = L / ne

    k = np.zeros((nn, nn))
    f = np.zeros(nn)

    for e in range(ne):
        Ke = element_stiffness(EA, Le)
        Fe = element_force(n0, Le)


        nodes = [e, e+1]

        for i in range(2):
            for j in range(2):
                k[nodes[i], nodes[j]] += Ke[i, j]
            f[nodes[i]] += Fe[i]

    K_red = k[1:, 1:]
    F_red = f[1:] 

    u_red = np.linalg.solve(K_red, F_red)


    u = np.zeros(nn)
    u[1:] = u_red


    x = np.linspace(0, L, nn)


    return x, u, f 


def analytical_u(x):
    return (n0 / (2 * EA)) * x * (2 * L - x)

x_exact = np.linspace(0, L, 200)
u_exact = analytical_u(x_exact)

plt.figure()
plt.plot(x_exact, u_exact, 'k--', label="Analytical")


for ne in [1, 3, 5, 10]:
    x, u, f = fem_solver(ne)
    plt.plot(x, u, marker='o', label=f"FEM ({ne} elements)")

plt.xlabel("x")
plt.ylabel("Displacement u")
plt.title("Displacement u(x)")
plt.legend()
plt.grid(True)
plt.show()


plt.figure()

for ne in [1, 3, 5, 10]:
    x, u, f = fem_solver(ne)
    plt.plot(x, f, marker='o', label=f"FEM ({ne} elements)")

plt.xlabel("x")
plt.ylabel("Nodal force F")
plt.title("Nodal external force distribution")
plt.legend()
plt.grid(True)
plt.show()