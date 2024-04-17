import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

N = 32
xx = np.linspace(0, 1, N + 1)
h = 1 / N


def f(x):
    return -(np.pi**2) * np.sin(np.pi * x)


A = -2 * np.eye(N - 1) + np.eye(N - 1, k=1) + np.eye(N - 1, k=-1)


def loss(u):
    res = A @ u - h**2 * f(xx[1:-1])
    return (res**2).mean()


sol = minimize(loss, np.ones((N - 1,)), method="BFGS", tol=1e-8)

uu = np.sin(np.pi * xx)
u = np.hstack([0.0, sol.x, 0.0])
print(f"success? {sol.success}")
plt.plot(xx, uu, label="ground truth")
plt.plot(xx, u, "--", label="found")
plt.title(f"relative err: {abs(u - uu).sum() / abs(uu).sum():.3e}")
plt.legend()
plt.tight_layout()
plt.savefig("figures/1")
