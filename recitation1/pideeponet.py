"""
Poissson equation, PI-DeepONet
    x in [0, 1]
    u_xx = -a * pi^2 sin(pi x)
    u(0) = u(1) = 0
    u(x) = a * sin(pi x)
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


def f(x, a):
    return -(jnp.pi**2) * jnp.sin(jnp.pi * x) * a


def u(x, a):
    return a * np.sin(np.pi * x)


init_br, apply_br = MLP([1, 64, 64])
init_tr, apply_tr = MLP([1, 64, 64])


def tr(params_t, x):
    return x * (1 - x) * apply_tr(params_t, jnp.atleast_1d(x)).squeeze()


def br(params_b, a):
    return apply_br(params_b, jnp.atleast_1d(a)).squeeze()


def onet(params, x, a):
    params_t, params_b = params
    t = tr(params_t, x)
    b = br(params_b, a)
    return jnp.dot(t, b)


def onet_x(params, x, a):
    return jax.jacfwd(onet, 1)(params, x, a)


def onet_xx(params, x, a):
    return jax.jacfwd(onet_x, 1)(params, x, a)


def _loss(params, x, a):
    return onet_xx(params, x, a) - f(x, a)


def loss(params, xx, aa):
    pde = jax.vmap(_loss, in_axes=(None, 0, 0))(params, xx, aa)
    loss = (pde**2).mean()
    return loss


nIter = 10000
lr = optax.cosine_decay_schedule(1e-3, nIter)
opt = optax.adam(lr)

# initialize
rng, key_b, key_t = jr.split(jr.PRNGKey(0), 3)
params_t = init_tr(key_t)
params_b = init_br(key_b)
params = (params_t, params_b)
state = opt.init(params)
x = np.linspace(0, 1, 300)
a = np.linspace(-1, 1, 100)
mesh = np.meshgrid(x, a)
xx = mesh[0].ravel()
aa = mesh[1].ravel()


@jax.jit
def step(params, state, xx, aa):
    v, g = jax.value_and_grad(loss)(params, xx, aa)
    updates, state = opt.update(g, state, params)
    params = optax.apply_updates(params, updates)
    return params, state, v


loss_total = []
for it in (pbar := trange(1, 1 + nIter)):
    params, state, v = step(params, state, xx, aa)
    if it % 100 == 0:
        loss_total.append(v)
        pbar.set_postfix({"loss": f"{v:.3e}"})


_, [(ax00, ax01), (ax10, ax11)] = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
# reference sol, prediction
x = np.linspace(0, 1, 500)
a = np.linspace(-1, 1, 300)
mesh = np.meshgrid(x, a, indexing="ij")
xx = mesh[0].ravel()
aa = mesh[1].ravel()
uu = u(*mesh)
upred = jax.vmap(onet, in_axes=(None, 0, 0))(params, xx, aa).reshape(500, -1)
err = np.abs(upred - uu).sum(0) / np.abs(uu).sum(0)

ax00.semilogy(loss_total, label=r"$\mathcal{L}_\mathrm{PINN}$")
ax00.legend()
ax00.set_title("Physics-informed DeepONet")
ax01.semilogy(a, err, label="relative error")
ax01.set_xlabel(r"$a$")
ax01.set_title(r"relative $L^1$ as a function of $a$")

uu = uu.ravel()
upred = upred.ravel()
ax10.tricontourf(xx, aa, uu)
ax10.set_title(r"$u$")

ax11.tricontourf(xx, aa, upred)
ax11.set_title(
    r"$u_\theta$, " + f"relative err: {np.abs(upred - uu).sum() / np.abs(uu).sum():.3e}"
)

plt.tight_layout()
plt.savefig("figures/pideeponet", dpi=100)
