import matplotlib.pyplot as plt
import numpy as np

N = 64
xx = np.linspace(0, 1, N + 1)
h = 1 / N


def f(x):
    return -(np.pi**2) * np.sin(np.pi * x)


A = -2 * np.eye(N - 1) + np.eye(N - 1, k=1) + np.eye(N - 1, k=-1)
b = h**2 * f(xx[1:-1])
u = np.linalg.solve(A, b)
u = np.hstack([0.0, u, 0.0])
uu = np.sin(np.pi * xx)


plt.plot(xx, uu, label="ground truth")
plt.plot(xx, u, "--", label="found")
plt.title(f"relative err: {abs(u - uu).sum() / abs(uu).sum():.3e}")
plt.legend()
plt.tight_layout()
plt.savefig("figures/0")
