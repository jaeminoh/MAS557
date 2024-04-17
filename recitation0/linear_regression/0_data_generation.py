import numpy as np

print("generating data...")

x = np.random.random(size=(100,))
eps = 0.1 * np.random.random(size=(100,))
y = 3 * x - 1 + eps
np.savez("data/data", x=x, y=y)
print("done!")
