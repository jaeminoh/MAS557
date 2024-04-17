import numpy as np

print("least square, direct solve")

data = np.load("data/data.npz", allow_pickle=True)
x, y = data["x"], data["y"]

x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

X = np.concatenate([x, np.ones(x.shape)], 1)

# X: (100, 2) matrox; y: (100, 1) column vector.
# normal eqquation: X.T @ X [weight, bias].T = X.T @ y

w_lstsq = np.linalg.lstsq(X.T @ X, X.T @ y, rcond=None)[0]
params = w_lstsq.ravel()

print(f"params: {params}")
