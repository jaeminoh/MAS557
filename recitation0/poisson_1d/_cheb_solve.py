import numpy as np
import matplotlib.pyplot as plt
from cheb import cheb


N = 2**5

D, xx = cheb(N)
L = D @ D
L = L[1:-1, 1:-1]


def f(x):
    return -(np.pi**2) * np.sin(np.pi * x)


sol = np.linalg.solve(L, f(xx[1:-1]))

uu = np.sin(np.pi * xx)
u = np.hstack([0.0, sol, 0.0])
plt.plot(xx, uu, label="ground truth")
plt.plot(xx, u, "--", label="found")
plt.title(f"relative err: {abs(u - uu).sum() / abs(uu).sum():.3e}")
plt.legend()
plt.tight_layout()
plt.savefig("figures/7")
