import jax
import jax.numpy as jnp
import jax.random as jr
import jaxopt
import matplotlib.pyplot as plt
import numpy as np
import optax
from tqdm import trange

from nn import MLP

init_tr, apply_tr = MLP([2, 200, 200, 200])  # t, x
init_br, apply_br = MLP([64, 200, 200, 200])  # 64 sensors


def onet(params, t, x, u):
    params_t, params_b = params
    tr = apply_tr(params_t, jnp.stack([t, x])) * (1 - x**2) * t  # zero boundary
    br = apply_br(params_b, u)
    onet = jnp.dot(tr, br)
    return onet


def onet_t(params, t, x, u):
    onet_t = jax.jacfwd(onet, 1)(params, t, x, u)
    return onet_t


def onet_x(params, t, x, u):
    onet_x = jax.jacfwd(onet, 2)(params, t, x, u)
    return onet_x


def onet_xx(params, t, x, u):
    onet_xx = jax.jacfwd(onet_x, 2)(params, t, x, u)
    return onet_xx


def vectorize(fn):
    v1 = jax.vmap(fn, (None, 0, None, None))
    v2 = jax.vmap(v1, (None, None, 0, None))
    v3 = jax.vmap(v2, (None, None, None, 0))
    return v3  # (tt, xx, uu) -> (#uu, #xx, #tt) array.


v_onet = vectorize(onet)
v_onet_t = vectorize(onet_t)
v_onet_xx = vectorize(onet_xx)


def loss(params, tt, xx, uu):
    s = v_onet(params, tt, xx, uu)
    s_t = v_onet_t(params, tt, xx, uu)
    s_xx = v_onet_xx(params, tt, xx, uu)
    loss = jnp.mean((s_t - 1e-2 * s_xx - 1e-2 * s - uu[..., None]) ** 2)
    return loss


tt = jnp.linspace(0, 1, 101)
xx = np.load("data/coordinate_x.npy", allow_pickle=True).squeeze()
rng = jr.key(1234)
rng, key1, key2 = jr.split(rng, 3)
w1 = jr.uniform(key1, (500, 1), maxval=jnp.pi * 2)
w2 = jr.uniform(key2, (500, 1), maxval=jnp.pi * 2)
uu = jnp.cos(w1 * xx) + jnp.sin(w2 * xx)  # (500, 64)

# initialize
key_t, key_b = jr.split(rng)
params = [init_tr(key_t), init_br(key_b)]
# adam
nIter = 10000
lr = optax.cosine_decay_schedule(1e-3, nIter)
opt = jaxopt.OptaxSolver(loss, optax.adamw(lr), maxiter=nIter)
state = opt.init_state(params, tt, xx, uu)


@jax.jit
def step(params, state):
    params, state = opt.update(params, state, tt, xx, uu)
    return params, state, state.value


# training

loss_total = []
for it in (pbar := trange(1, 1 + nIter)):
    params, state, v = step(params, state)
    if it % 100 == 0:
        loss_total.append(v)
        pbar.set_postfix({"loss": f"{v:.3e}"})

# visualize
xx = np.load("data/coordinate_x.npy", allow_pickle=True).squeeze()

data = np.load("data/solution.npy", allow_pickle=True)
uu = data[:, :, -1]  # (50, 64)
ss = data[:, :, :-1]  # (50, 64, 101) = (#sol, xx, tt)

pred = v_onet(params, tt, xx, uu)


@jax.vmap
def rel_l2(pred, true):
    return jnp.linalg.norm(pred - true) / jnp.linalg.norm(true)


err = rel_l2(pred, ss)

_, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(12, 4))
ax0.semilogy(err)
ax0.set_xlabel("index")
ax0.set_title("relative L2 error")
ax1.imshow(ss[1], cmap="jet", aspect="auto")
ax1.set_title("true")
ax2.imshow(pred[1], cmap="jet", aspect="auto")
ax2.set_title("pred")
plt.tight_layout()
plt.savefig("figures/onet", dpi=100)
