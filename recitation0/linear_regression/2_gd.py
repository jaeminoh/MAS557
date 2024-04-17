import numpy as np
from tqdm import trange

print("gradient descent, numpy")
data = np.load("data/data.npz", allow_pickle=True)
x, y = data["x"], data["y"]

# initial guess for weight and bias
params = np.array([0.0, 0.0])


def loss(params):
    weight, bias = params
    loss = ((y - weight * x - bias) ** 2).mean()
    return loss


def grad(params):
    weight, bias = params
    dl_dw = (-2 * (y - weight * x - bias) * x).mean()
    dl_db = (-2 * (y - weight * x - bias)).mean()
    return np.array([dl_dw, dl_db])


def step(params, lr=0.1):
    g = grad(params)
    params = params - lr * g
    return params


for it in (pbar := trange(1000)):
    params = step(params)
    if it % 100 == 0:
        v = loss(params)
        pbar.set_postfix({"loss": f"{v:.3e}"})

print(f"params: {params}")
