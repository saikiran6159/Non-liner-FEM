import matplotlib.pyplot as plt

A = 9
x = 1
tol = 1e-5
max_iter = 100

x_val = []
y_val = []
n_val = []

for n in range(max_iter):
    y = A/x
    x_new = 0.5 * (x + y)

    x_val.append(x)
    y_val.append(y)
    n_val.append(n)

    if abs(x_new - x) < tol:
        break

    x = x_new

print("square root of A =", x_new)
print("iteration:", n+1)


plt.plot(n_val, x_val, label="x_n")
plt.plot(n_val, y_val, label="y_n")
plt.xlabel("iterations")
plt.ylabel("values")
plt.title("babylonian method")
plt.legend()
plt.grid(True)
plt.show()


