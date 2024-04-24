"""
Boundary Layer 1d
    x in [0, 1]
    -eps u_xx - u_x = 1
    u(0) = u(1) = 0
    u(x) = -x - (1 - e^(-x / eps)) / (1 - e^(-1 / eps))
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import jaxopt
import matplotlib.pyplot as plt
import numpy as np
import time

from nn import MLP

jax.config.update("jax_enable_x64", True)


seed = jr.PRNGKey(2)  # 2: correct; 3: opposite
eps = 1e-6


def u(x):
    u = -x + (1 - jnp.exp(-x / eps)) / (1 - jnp.exp(-1 / eps))
    return u


key, subkey = jr.split(seed)
init_params, apply = MLP([1, 20, 20, 20, 1], jnp.tanh)
params = init_params(subkey)


def pinn(params, x):
    pinn = apply(params, jnp.atleast_1d(x)).squeeze()
    pinn = x * (1 - x) * pinn
    return pinn


def pinn_x(params, x):
    pinn_x = jax.jacfwd(pinn, 1)(params, x)
    return pinn_x


def pinn_xx(params, x):
    pinn_xx = jax.jacfwd(pinn_x, 1)(params, x)
    return pinn_xx


def pde(params, x):
    pde = -eps * pinn_xx(params, x) - pinn_x(params, x) - 1
    return pde


def loss(params, x):
    res = jax.vmap(pde, in_axes=(None, 0))(params, x)
    loss = (res**2).mean()
    return loss


x = jnp.linspace(0, 1, 300)
opt = jaxopt.LBFGS(loss, history_size=50)
state = opt.init_state(params, x)


@jax.jit
def step(params, state, x):
    params, state = opt.update(params, state, x)
    return params, state


loss_traj = []
print("LBFGS running...")
tic = time.time()
for it in range(1, 10000 + 1):
    params, state = step(params, state, x)
    if it % 500 == 0:
        pinn_loss = state.value
        loss_traj.append(pinn_loss)
        print(f"it: {it}, loss: {pinn_loss:.3e}")
toc = time.time()
print(f"Done! Elapsed time: {toc - tic:.2f}")


x_test = np.linspace(0, 1, 3000)
u_pred = jax.vmap(pinn, in_axes=(None, 0))(params, x_test)
u_true = u(x_test)

plt.plot(x_test, u_true, label=r"$u$")
err = jnp.abs(u_pred - u_true).sum() / jnp.abs(u_true).sum()
plt.plot(x_test, u_pred, "--", label=r"$u_\theta$")
plt.title(r"$\epsilon=$" + f"{eps:.2e}, " f"err: {err:.3e}")
plt.legend()
plt.tight_layout()
plt.savefig("figures/singular1d_result", dpi=100)

plt.cla()
plt.semilogy(loss_traj)
plt.title("PINN Loss")
plt.tight_layout()
plt.savefig("figures/singular1d_loss", dpi=100)
