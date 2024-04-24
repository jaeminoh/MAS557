"""
2d Poisson Equation, PINN
    domain: "L" shaped region (-1, 1)^2 setminus (0, 1)^2
    Laplacian[u] = -1
    Zero Dirichlet Boundary condition
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import jaxopt
import time
from scipy.stats.qmc import Sobol

from nn import MLP

jax.config.update("jax_enable_x64", True)


# interior point sampling
def sampling_interior(m: int = 9):
    sobol = Sobol(d=2)
    # sampling 3 * 2^m points
    xy1 = sobol.random_base2(m)
    sobol.reset()
    xy2 = sobol.random_base2(m)
    sobol.reset()
    xy3 = sobol.random_base2(m)
    sobol.reset()
    xy1[:, 0] = xy1[:, 0] - 1.0
    xy2 = xy2 - 1
    xy3[:, 1] = xy3[:, 1] - 1.0
    xy_interior = jnp.concatenate([xy1, xy2, xy3])
    return xy_interior


# boundary point sampling
def sampling_boundary(m: int = 8):
    sobol = Sobol(d=1)
    # sampling 6 * 2^m points
    N = 2**m
    # x
    x1 = jnp.stack([sobol.random_base2(m).squeeze() * 2 - 1, -1 * jnp.ones((N,))], 1)
    sobol.reset()
    x2 = jnp.stack([sobol.random_base2(m).squeeze() - 1, jnp.ones((N,))], 1)
    sobol.reset()
    x3 = jnp.stack([sobol.random_base2(m).squeeze(), jnp.zeros((N,))], 1)
    sobol.reset()
    # y
    y1 = jnp.stack(
        [
            -1 * jnp.ones((N,)),
            sobol.random_base2(m).squeeze() * 2 - 1,
        ],
        1,
    )
    sobol.reset()
    y2 = jnp.stack([jnp.zeros((N,)), sobol.random_base2(m).squeeze()], 1)
    sobol.reset()
    y3 = jnp.stack([jnp.ones((N,)), sobol.random_base2(m).squeeze() - 1], 1)
    sobol.reset()
    xy_boundary = jnp.concatenate([x1, x2, x3, y1, y2, y3])
    return xy_boundary


xy_in = sampling_interior()
xy_b = sampling_boundary()


init, apply = MLP([2, 50, 50, 50, 50, 1], jnp.tanh)


def pinn(params, x, y):
    inputs = jnp.stack([x, y])
    pinn = apply(params, inputs).squeeze()  # scalar
    return pinn


def pinn_x(params, x, y):
    return jax.jacfwd(pinn, 1)(params, x, y)


def pinn_xx(params, x, y):
    return jax.jacfwd(pinn_x, 1)(params, x, y)


def pinn_y(params, x, y):
    return jax.jacfwd(pinn, 2)(params, x, y)


def pinn_yy(params, x, y):
    return jax.jacfwd(pinn_y, 2)(params, x, y)


def pde(params, xy_in):
    x, y = xy_in
    u_xx = pinn_xx(params, x, y)
    u_yy = pinn_yy(params, x, y)
    return u_xx + u_yy + 1


def bc(params, xy_b):
    x, y = xy_b
    u = pinn(params, x, y)
    return u


def loss(params, xy_in, xy_b):
    pde_res = jax.vmap(pde, in_axes=(None, 0))(params, xy_in)
    bc_res = jax.vmap(bc, in_axes=(None, 0))(params, xy_b)
    pde_loss = (pde_res**2).mean()
    bc_loss = (bc_res**2).mean()
    return pde_loss + 1e2 * bc_loss, (pde_loss, bc_loss)


nIter = 10000
opt = jaxopt.LBFGS(loss, has_aux=True)

# initialize
params = init(jr.PRNGKey(0))
state = opt.init_state(params, xy_in, xy_b)


@jax.jit
def step(params, state, xy_in=xy_in, xy_b=xy_b):
    params, state = opt.update(params, state, xy_in, xy_b)
    return params, state


loss_total = []
loss_pde = []
loss_bc = []
print("Solving...")
tic = time.time()
for it in range(1, 1 + 10000):
    params, state = step(params, state)
    if it % 100 == 0:
        total_loss = state.value
        pde_loss, bc_loss = state.aux
        loss_total.append(total_loss)
        loss_pde.append(pde_loss)
        loss_bc.append(bc_loss)
        print(f"it: {it}, loss: {total_loss:.3e}")
toc = time.time()
print(f"Done! Elapsed time: {toc - tic:.2f}")

_, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8, 4))
ax0.semilogy(loss_total, label=r"$\mathcal{L}_\mathrm{PINN}$")
ax0.semilogy(loss_pde, "--", label=r"$\mathcal{L}_\mathrm{pde}$")
ax0.semilogy(loss_bc, ":", label=r"$\mathcal{L}_\mathrm{bc}$")
ax0.legend()
ax0.set_title("PINN")

x, y = xy_in[:, 0], xy_in[:, 1]
u_pred = jax.vmap(pinn, (None, 0, 0))(params, x, y)
ax1.tricontourf(x, y, u_pred, cmap="jet")
ax1.set_title(r"$u_\theta$")
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$y$")

plt.tight_layout()
plt.savefig("figures/poisson2d", dpi=100)
