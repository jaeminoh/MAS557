import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

N = 64
xx = np.linspace(0, 1, N + 1)
h = 1 / N


def f(x):
    return -(np.pi**2) * np.sin(np.pi * x)


A = -2 * np.eye(N - 1) + np.eye(N - 1, k=1) + np.eye(N - 1, k=-1)

eig, _ = np.linalg.eig(A / h**2)
eig = abs(eig)
print(eig.max() / eig.min())


"""@jax.value_and_grad
def loss(u):
    res = jnp.convolve(u, jnp.array([1., -2., 1.]), mode='same') - h**2 * f(xx[1:-1])
    return (res**2).mean()


@jax.jit
def step(u):
    l, g = loss(u)
    u = u - 0.01 * g
    return u, l


u = np.ones((N - 1,))

for it in (pbar := (trange(100000))):
    u, l = step(u)
    if it % 100 == 0:
        pbar.set_postfix({"loss": f"{l:.3e}"})

uu = np.sin(np.pi * xx)
u = np.hstack([0.0, u, 0.0])
plt.plot(xx, uu, label="ground truth")
plt.plot(xx, u, "--", label="found")
plt.title(f"relative err: {abs(u - uu).sum() / abs(uu).sum():.3e}")
plt.legend()
plt.tight_layout()
plt.savefig("figures/2")"""
