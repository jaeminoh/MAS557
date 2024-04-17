import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

N = 2**8
xx = np.linspace(0, 1, N + 1)
h = 1 / N


def f(x):
    return -(np.pi**2) * np.sin(np.pi * x)


def loss(u):
    res = jnp.convolve(u, jnp.array([1.0, -2.0, 1.0]), mode="same") / h**2 - f(xx[1:-1])
    return (res**2).mean()


opt = jaxopt.BFGS(loss, maxiter=1000, tol=1e-20)
initial_guess = np.ones((N - 1,)) * 0.5
sol, state = opt.run(initial_guess)

uu = np.sin(np.pi * xx)
u = np.hstack([0.0, sol, 0.0])
plt.plot(xx, uu, label="ground truth")
plt.plot(xx, u, "--", label="found")
plt.title(f"relative err: {abs(u - uu).sum() / abs(uu).sum():.3e}")
plt.legend()
plt.tight_layout()
plt.savefig("figures/5")
