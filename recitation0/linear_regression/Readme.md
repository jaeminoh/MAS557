# Linear regression

## Problem

$y = 3x -1 + \epsilon, \quad \epsilon\sim N(0, 0.1^2)$.

Suppose that the coefficient $3$ and the intercept $-1$ is unknown.
We aim to find 
$$(\hat w, \hat b) = \argmin_{w, b} \sum_{i=1}^N (y_i - wx_i - b)^2.$$

## Code

```0_data_generation.py``` generates 100 random samples $\{(x_i, y_i)_{i=1}^{100}\}$,
where 
$$x_i \sim_{iid} N(0, 1), \quad \epsilon_i \sim_{iid} N(0, 0.1^2), \quad y_i = 3 x_i - 1 + \epsilon_i.$$

```1_lstsq.py``` finds optimal parameters by solving the normal equation
$$X^T X \beta = X^T Y$$
with ```numpy.linalg.lstsq```.

```2_gd.py``` finds parameters by 1000 gradient descent steps, with hand-calculated gradients.
It is a pure ```numpy``` code.

```3_gd_ad.py``` takes 1000 gradient descent steps, by calculating gradients with automatic differentiation algorithm provided in JAX.
Observe that this is slower than ```2_gd.py```.

```4_gd_ad_jit.py``` is almost same to ```3_gd_ad.py```,
but the function ```step``` is decorated by ```jax.jit```.
This decoration compiles ```step``` function, which is the most expensive routine of the graddient descent,
therefore we have significant speed-ups.
(This is one of the key features of JAX.)

Lastly, ```5_mlp.py``` finds network parameters $\hat \theta$ which minimizes
$$ \frac{1}{100}\sum_{i=1}^{100} (y_i - \mathrm{MLP}_\theta (x_i))^2. $$

Note that, we may compile ```2_gd.py``` by using ```numba``` package to enjoy further speed-ups.
