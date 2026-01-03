import matplotlib.pyplot as plt
import math

A = 9
tol = 1e-10
max_iter = 50

# function and derivative
def f(x):
    return x*x - A

def df(x):
    return 2*x

# -----------------------------
# a) False position method
# -----------------------------
a = 1.0
b = 5.0

x_false = []
err_false = []

for k in range(max_iter):
    x_new = b - f(b)*(b-a)/(f(b)-f(a))
    x_false.append(x_new)

    if k > 0:
        err_false.append(abs(x_false[k] - x_false[k-1]))

    if f(a)*f(x_new) < 0:
        b = x_new
    else:
        a = x_new

    if k > 0 and err_false[-1] < tol:
        break

# -----------------------------
# b) Newton method
# -----------------------------
x = 1.0
x_newton = []
err_newton = []

for k in range(max_iter):
    x_new = x - f(x)/df(x)
    x_newton.append(x_new)

    if k > 0:
        err_newton.append(abs(x_newton[k] - x_newton[k-1]))

    if k > 0 and err_newton[-1] < tol:
        break

    x = x_new

# -----------------------------
# Semi-log plot
# -----------------------------
plt.semilogy(err_false, label="False position")
plt.semilogy(err_newton, label="Newton method")

plt.xlabel("Iteration k")
plt.ylabel("|x(k+1) - x(k)|")
plt.title("Convergence comparison for √A")
plt.legend()
plt.grid(True, which="both")
plt.show()
