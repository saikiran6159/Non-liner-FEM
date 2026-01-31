#imports

import numpy as np
import matplotlib.pyplot as plt

#problem parameters mat no. 69297 

a = 40.0
b = 80.0
E = 70000.0
nu = 0.25
Q = 35000.0
T = 1.0
p_max = 50.0
t_L = 2.0
t_F = 10.0
tol = 5e-3
max_iter = 25

# now the materials parameters
 
def elastic_stiffness(E, nu):
    fac = E /((1 + nu)*(1 - 2*nu)) 
    return fac * np.array([
        [1-nu, nu, nu],
        [nu, 1-nu, nu],
        [nu, nu, 1-nu]
    ])

def P_dev():
    return np.array([
        [2/3, -1/3, -1/3],
        [-1/3, 2/3, -1/3],
        [-1/3, -1/3, 2/3]
    ])

def pressure(t):
    if t <= t_L:
        return p_max * t/t_L
    else:
        return p_max
    

# element routine for visco-elastic

def element_viscoelastic(r_e, u_e, eps_n, s_n, dt, Q_use):
    # conside gauss one point because linear 

    w = 2.0
    N = np.array([0.5, 0.5])
    dN_dxi = np.array([-0.5, 0.5])
    J = (r_e[1] - r_e[0])/2
    dN_dr = dN_dxi/J
    r_gp = N @ r_e

    B = np.array([
        [dN_dr[0], dN_dr[1]],
        [N[0]/r_gp, N[1]/r_gp],
        [0.0, 0.0]
    ])

    eps = B @ u_e
    deps = eps - eps_n

    P = P_dev()
    deps_dev = P @ deps

    s_np1 = (T / (T + dt)) * s_n + (T * Q_use / (T + dt)) * deps_dev
    
    C = elastic_stiffness(E, nu)
    sigma = (C @ eps) + s_np1

    C_tan = C + (Q_use * T/(T + dt)) * P

    F_e = B.T @ sigma * r_gp * J * w
    K_e = B.T @ C_tan @ B * r_gp * J * w

    return F_e, K_e, eps, s_np1


#newton_solver

def newton_solver(dt, ne, Q_use):

    nodes = np.linspace(a, b, ne+1)
    elements = [(i, i+1) for i in range(ne)]
    ndof = len(nodes)
    
    u = np.zeros(ndof)
    s_e = np.zeros((ne, 3))
    eps_e = np.zeros((ne, 3))

    u_history = []
    time_history = []
    it_history = []
    time = np.arange(0.0, t_F + dt, dt)
    for n in range(1, len(time)):
        t = time[n]
        p = pressure(t)
        u_iter = u.copy()

        for it in range(max_iter):
            K = np.zeros((ndof, ndof))
            F_int = np.zeros(ndof)

            for e, (i, j) in enumerate(elements):

                r_e = nodes[[i, j]]
                u_e = u_iter[[i, j]]

                F_e, K_e, _, _ = element_viscoelastic(r_e, u_e, eps_e[e], s_e[e], dt, Q_use)

                F_int[i:j+1] += F_e
                K[i:j+1, i:j+1] += K_e

            F_ext = np.zeros(ndof)
            F_ext[0] = -p * a 

            R = F_int - F_ext

            du = np.linalg.solve(K, -R)
            u_iter += du

            conv_force = np.linalg.norm(R, np.inf)/max(np.linalg.norm(F_int,np.inf), 1.0) < tol
            conv_disp = np.linalg.norm(du, np.inf)/max(np.linalg.norm(u_iter, np.inf), 1.0) < tol

            if conv_force and conv_disp:
                break
        it_history.append(it + 1)
        u = u_iter.copy()

        for e, (i, j) in enumerate(elements):
            r_e = nodes[[i, j]]
            u_e = u[[i, j]]
            _, _, eps_e[e], s_e[e] =  element_viscoelastic(r_e, u_e, eps_e[e], s_e[e], dt, Q_use)

        u_history.append(u.copy())
        time_history.append(t)

    return u_history, time_history, nodes, it_history


u_hist, t_hist, nodes, it_hist = newton_solver(dt=0.1, ne=20, Q_use=Q)


#displacement vs radius

plt.figure()
for idx in [0, int(t_L/0.1), -1]:
    plt.plot(nodes, u_hist[idx], label=f"t = {t_hist[idx]:.1f}")
plt.xlabel("Radius r")
plt.ylabel("Radial displacement u_r")
plt.legend()
plt.grid()
plt.show()


#displacement vs time at inner radius 

u_inner = [u[0] for u in u_hist]
plt.figure()
plt.plot(t_hist, u_inner)
plt.xlabel("Time")
plt.ylabel("u_r at inner radius")
plt.grid()
plt.show()

ne_list = [5, 10, 20, 40]
u_mesh = []

for ne in ne_list:
    u_hist, _, _, _= newton_solver(dt=0.1, ne=ne, Q_use=Q)
    u_mesh.append(u_hist[-1][0])

plt.figure()
plt.plot(ne_list, u_mesh, marker='o')
plt.xlabel("Number of elements")
plt.ylabel("Final u_r at inner radius")
plt.grid()
plt.show()


# =====================================================
# Time-step convergence study
# =====================================================

dt_list = [0.5, 0.25, 0.1]
u_time = []

for dt in dt_list:
    u_hist, _, _, _ = newton_solver(dt=dt, ne=20, Q_use=Q)
    u_time.append(u_hist[-1][0])

plt.figure()
plt.plot(dt_list, u_time, marker='o')
plt.xlabel("Time step Δt")
plt.ylabel("Final u_r at inner radius")
plt.grid()
plt.show()

u_hist, t_hist, nodes, it_hist = newton_solver(dt=0.1, ne=20, Q_use=0.0)

r = nodes
u_exact = - (1 + nu) * (p_max / E) * (a**2 / (b**2 - a**2)) * (
    (1 - 2*nu) * r + b**2 / r
)

plt.figure()
plt.plot(r, u_hist[-1], 'o-', label="FEM (elastic)")
plt.plot(r, u_exact, '--', label="Analytical")
plt.xlabel("r")
plt.ylabel("u_r")
plt.legend()
plt.grid()
plt.show()
