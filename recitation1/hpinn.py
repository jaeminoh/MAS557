"""
Poissson equation, hPINN
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


def pinn_xx(params, x):
    return jax.jacfwd(pinn_x, 1)(params, x)


def _loss(params, x):
    u_xx = pinn_xx(params, x)
    pde = u_xx - f(x)
    bc = jnp.stack([pinn(params, 0.0), pinn(params, 1.0)])
    return pde, bc


def loss(params, xx):
    pde, bc = jax.vmap(_loss, in_axes=(None, 0))(params, xx)
    pde = (pde**2).mean()
    bc = (bc**2).mean()
    loss = pde + bc
    return loss, [pde, bc]


nIter = 10000
lr = optax.cosine_decay_schedule(1e-3, nIter)
opt = optax.adam(lr)

# initialize
params = init(jr.PRNGKey(0))
state = opt.init(params)
xx = np.linspace(0, 1, 300)


@jax.jit
def step(params, state, xx):
    v, g = jax.value_and_grad(loss, has_aux=True)(params, xx)
    updates, state = opt.update(g, state, params)
    params = optax.apply_updates(params, updates)
    return params, state, v


loss_total = []
loss_pde = []
loss_bc = []
for it in (pbar := trange(1, 1 + nIter)):
    params, state, v = step(params, state, xx)
    if it % 100 == 0:
        t, (pde, bc) = v
        loss_total.append(t)
        loss_pde.append(pde)
        loss_bc.append(bc)
        pbar.set_postfix({"loss": f"{t:.3e}"})


_, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8, 4))
ax0.semilogy(loss_total, label=r"$\mathcal{L}_\mathrm{PINN}$")
ax0.semilogy(loss_pde, "--", label=r"$\mathcal{L}_\mathrm{pde}$")
ax0.semilogy(loss_bc, ":", label=r"$\mathcal{L}_\mathrm{bc}$")
ax0.legend()
ax0.set_title("hPINN")

xx = np.linspace(0, 1, 1000)
uu = u(xx)
ax1.plot(xx, uu, label=r"$u$")
upred = jax.vmap(pinn, in_axes=(None, 0))(params, xx)
ax1.plot(xx, upred, "--", label=r"$u_\theta$")
ax1.set_title(f"relative err: {np.abs(upred - uu).sum() / np.abs(uu).sum():.3e}")

plt.tight_layout()
plt.savefig("figures/hpinn", dpi=100)
