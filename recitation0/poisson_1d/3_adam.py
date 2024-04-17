import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from tqdm import trange

jax.config.update("jax_enable_x64", True)

N = 2**6
xx = np.linspace(0, 1, N + 1)
h = 1 / N


def f(x):
    return -(np.pi**2) * np.sin(np.pi * x)


@jax.value_and_grad
def loss(u):
    res = jnp.convolve(u, jnp.array([1.0, -2.0, 1.0]), mode="same") / h**2 - f(
        xx[1:-1]
    )  # scalable than forming matrix
    return (res**2).mean()


nIter = 10**6
lr = optax.exponential_decay(1e-1, 1000, 0.91)
opt = optax.adam(lr)


@jax.jit
def step(u, state):
    v, g = loss(u)
    updates, state = opt.update(g, state, u)
    u = optax.apply_updates(u, updates)
    return u, state, v


u = np.ones((N - 1,))
state = opt.init(u)

for it in (pbar := (trange(nIter))):
    u, state, v = step(u, state)
    if it % 100 == 0:
        pbar.set_postfix({"loss": f"{v:.3e}"})

uu = np.sin(np.pi * xx)
u = np.hstack([0.0, u, 0.0])
plt.plot(xx, uu, label="ground truth")
plt.plot(xx, u, "--", label="found")
plt.title(f"relative err: {abs(u - uu).sum() / abs(uu).sum():.3e}")
plt.legend()
plt.tight_layout()
plt.savefig("figures/3")
