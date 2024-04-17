import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

print("mlp regression.")
data = np.load("data/data.npz", allow_pickle=True)
x, y = data["x"], data["y"]


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


init, apply = MLP()


def model(params, x):
    x = x.reshape(-1, 1)
    y = apply(params, x)
    return y.squeeze()


params = init(jr.PRNGKey(0))


@jax.value_and_grad
def loss(params):
    y_pred = model(params, x)
    loss = ((y - y_pred) ** 2).mean()
    return loss


@jax.jit
def step(params, lr=0.1):
    v, g = loss(params)
    params = jax.tree_map(lambda p, u: p - lr * u, params, g)
    return params, v


for it in (pbar := trange(1000)):
    params, v = step(params)
    if it % 100 == 0:
        pbar.set_postfix({"loss": f"{v:.3e}"})

print(f"loss: {v:.3e}")
xx = np.linspace(x.min(), x.max(), 150)
yy = model(params, xx)

plt.plot(x, y, ".", label="data")
plt.plot(xx, yy, "--", label="pred")
plt.legend()
plt.tight_layout()
plt.savefig("figures/mlp")
