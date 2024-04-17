import jax
import numpy as np

from cheb import cheb

jax.config.update("jax_enable_x64", True)

N = 2**6

D, xx = cheb(N)
L = D @ D
L = L[1:-1, 1:-1]
print(L)

eig, _ = np.linalg.eig(L)
eig = abs(eig)
print(eig.max() / eig.min())


"""def f(x):
    return -(np.pi**2) * np.sin(np.pi * x)


def loss(u):
    res = L @ u - f(xx[1:-1])
    return (res**2).mean()

lr = optax.exponential_decay(1e-1, 10000, 0.91)
opt = optax.adam(lr)
opt = jaxopt.OptaxSolver(loss, opt=opt, maxiter=10000000, tol=1e-32)
initial_guess = np.ones((N - 1,))
tic = time.time()
sol, state = opt.run(initial_guess)
toc = time.time()
print(f"Elapsed time: {toc - tic:.2f}s")

uu = np.sin(np.pi * xx)
u = np.hstack([0.0, sol, 0.0])
plt.plot(xx, uu, label="ground truth")
plt.plot(xx, u, "--", label="found")
plt.title(f"relative err: {abs(u - uu).sum() / abs(uu).sum():.3e}")
plt.legend()
plt.tight_layout()
plt.savefig("figures/6")"""
