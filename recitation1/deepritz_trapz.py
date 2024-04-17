"""
Poissson equation, DeepRitz
    x in [0, 1]
    u_xx = -pi^2 sin(pi x)
    u(0) = u(1) = 0
    u(x) = sin(pi x)
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import optax


# define MLP
def MLP(layers: list[int] = [1, 64, 1], activation: callable = jnp.tanh):
    def init_params(key):
        def _init(key, d_in, d_out):
            w = jr.normal(key, shape=(d_in, d_out)) * np.sqrt(2 / (d_in + d_out))
            b = jnp.zeros((d_out,))
            return [w, b]

        keys = jr.split(key, len(layers) - 1)
        params = list(map(_init, keys, layers[:-1], layers[1:]))
        return params

    def apply(params, inputs):
        for W, b in params[:-1]:
            outputs = inputs @ W + b
            inputs = activation(outputs)
        W, b = params[-1]
        outputs = inputs @ W + b
        return outputs

    return init_params, apply


def f(x):
    return -(jnp.pi**2) * jnp.sin(jnp.pi * x)


def u(x):
    return np.sin(np.pi * x)


init, apply = MLP()


def pinn(params, x):
    return x * (1 - x) * apply(params, jnp.atleast_1d(x)).squeeze()


def pinn_x(params, x):
    return jax.jacfwd(pinn, 1)(params, x)


def loss(params, xx):
    h = xx[1] - xx[0]
    u = jax.vmap(pinn, in_axes=(None, 0))(params, xx)
    u_x = jax.vmap(pinn_x, in_axes=(None, 0))(params, xx)
    _energy = u_x**2
    _work = -f(xx) * u
    energy = (0.5 * (_energy[0] + _energy[-1]) + jnp.sum(_energy[1:-1])) * h
    work = (0.5 * (_work[0] + _work[-1]) + jnp.sum(_work[1:-1])) * h
    return 0.5 * energy - work


nIter = 10000
lr = optax.cosine_decay_schedule(1e-3, nIter)
opt = optax.adam(lr)

# initialize
params = init(jr.PRNGKey(0))
state = opt.init(params)
xx = np.linspace(0, 1, 100)


@jax.jit
def step(params, state, xx):
    v, g = jax.value_and_grad(loss)(params, xx)
    updates, state = opt.update(g, state, params)
    params = optax.apply_updates(params, updates)
    return params, state, v


loss_total = []
for it in (pbar := trange(1, 1 + nIter)):
    params, state, v = step(params, state, xx)
    if it % 100 == 0:
        loss_total.append(v)
        pbar.set_postfix({"loss": f"{v:.3e}"})


_, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8, 4))
ax0.plot(loss_total, label="Energy")
ax0.set_title("Deep Ritz")
ax0.legend()

xx = np.linspace(0, 1, 1000)
uu = u(xx)
ax1.plot(xx, uu, label=r"$u$")
upred = jax.vmap(pinn, in_axes=(None, 0))(params, xx)
ax1.plot(xx, upred, "--", label=r"$u_\theta$")
ax1.set_title(f"relative err: {np.abs(upred - uu).sum() / np.abs(uu).sum():.3e}")

plt.tight_layout()
plt.savefig("figures/deepritz_trapz", dpi=100)
