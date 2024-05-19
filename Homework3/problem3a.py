import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import jaxopt

from nn import MLP

jax.config.update("jax_enable_x64", True)

eps = 1e-3


def f(x):
    return 3 / eps * jnp.exp(-2 * x / eps)


def u(x):
    return 1.5 * (
        x - (np.exp(-2 * (1 - x) / eps) - np.exp(-2 / eps)) / (1 - np.exp(-2 / eps))
    )


init, apply = MLP([1, 50, 50, 50, 1])


def pinn(params, x):
    return x * (1 - x) * apply(params, jnp.atleast_1d(x)).squeeze()


def pinn_x(params, x):
    return jax.jacfwd(pinn, 1)(params, x)


def loss(params, xx):
    h = xx[1] - xx[0]
    u = jax.vmap(pinn, in_axes=(None, 0))(params, xx)
    u_x = jax.vmap(pinn_x, in_axes=(None, 0))(params, xx)
    _energy = jnp.exp(-2 * xx / eps) * u_x**2
    _work = f(xx) * u
    energy = (0.5 * (_energy[0] + _energy[-1]) + jnp.sum(_energy[1:-1])) * h
    work = (0.5 * (_work[0] + _work[-1]) + jnp.sum(_work[1:-1])) * h
    return 0.5 * energy - work


nIter = 5000
opt = jaxopt.LBFGS(loss)

# initialize
params = init(jr.PRNGKey(0))
xx = np.linspace(0, 1, 1000)
state = opt.init_state(params, xx)


@jax.jit
def step(params, state, xx):
    params, state = opt.update(params, state, xx)
    return params, state, state.value


loss_total = []
for it in (pbar := trange(1, 1 + nIter)):
    params, state, v = step(params, state, xx)
    if it % 100 == 0:
        loss_total.append(v)
        pbar.set_postfix({"loss": f"{v:.3e}"})


_, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8, 4))
ax0.plot(loss_total, label="Energy")
ax0.set_title(r"Deep Ritz, $\epsilon$: " + f"{eps:.2e}")
ax0.legend()

xx = np.linspace(0, 1, 500)
uu = u(xx)
ax1.plot(xx, uu, label=r"$u$")
upred = jax.vmap(pinn, in_axes=(None, 0))(params, xx)
ax1.plot(xx, upred, "--", label=r"$u_\theta$")
ax1.set_title(f"relative err: {np.abs(upred - uu).sum() / np.abs(uu).sum():.3e}")

plt.tight_layout()
plt.savefig(f"figures/deepritz_eps{eps}.png", format="png", dpi=100)
