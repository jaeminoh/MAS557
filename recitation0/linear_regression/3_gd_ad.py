import jax
import numpy as np
from tqdm import trange

print("gradient descent, AD, jax")
data = np.load("data/data.npz", allow_pickle=True)
x, y = data["x"], data["y"]

# initial guess for weight and bias
params = np.array([0.0, 0.0])


@jax.value_and_grad
def loss(params):
    weight, bias = params
    loss = ((y - weight * x - bias) ** 2).mean()
    return loss


def step(params, lr=0.1):
    v, g = loss(params)
    params = params - lr * g
    return params, v


for it in (pbar := trange(1000)):
    params, v = step(params)
    if it % 100 == 0:
        pbar.set_postfix({"loss": f"{v:.3e}"})

print(f"params: {params}")
