# Poisson Equation

## Equation
We consider the following equation:
$$ u_{xx}(x) = -\pi^2 \sin(\pi x), \quad x \in (0, 1), \quad u(0) = u(1) = 0.$$
Note that the exact solution is $u(x) = \sin(\pi x)$.

## Finite Difference Method
We discretize the domain $[0, 1]$ into $N+1$ equally-spaced points.
That is,
$$ (x_0, \dots, x_i, \dots, x_N) = \left(0, \dots, \frac{i}{N}, \dots, 1\right).$$
We introduce $N+1$ unknowns $\{u_i\}_{i=0}^N$, which is $u(x_i) = u_i$.
Note that $u_0 = u_N = 0$ because of the boundary condition.
So we actually introduced $N-1$ unknowns.

We evaluate the equation at $x_1, \dots, x_{N-1}$.
Then we have
$$u_{xx}(x_i) = -\pi^2 \sin(\pi x_i).$$
The left hand side can be further approximated by finite difference formula,
$$u_{xx}(x_i) \approx \frac{u_{i+1} - 2u_i + u_{i-1}}{h^2},$$
where $h = 1/N$.
With matrix notation, this becomes
$$A u = b$$
where $A$ is a tridiagonal matrix whose main diagonal is $-2$, and the upper and the lower diagonal entries are all $1$, and $b = -h^2 (f(x_1), \dots, f(x_{N-1}))^T$.

## Code
```0_lstsq.py``` solves $\hat u = A^{-1} b$ directly.

```1_bfgs.py``` minimizes a loss function $L(u) = \|A u - b\|_2^2$ with BFGS optimization algorithm.
Note that BFGS algorithm is a quasi-Newton method,
which approximates the inverse of the hessian matrix with histories of gradients.
We utilized ```scipy.optimize``` module,
but more convenient libraries (automatic differentiation for the jacobian matrix) are available, such as JAXopt, or optimistix.

```2_gd.py``` minimizes the loss function $L(u)$ by taking gradient descent steps, where gradient is obtained via automatic differentiation provided in JAX.

```3_adam.py``` uses adam optimizer.
Here, we used Optax library for adam.
Optax library contains a lot of first-order optimization methods (adam, adamW, SAM, sgd, lion, AdaGrad, ...).

```4_rootfinding.py``` finds a root of the residual function
$$[r(u)]_i = r_i(u) = u_{i+1} - 2u_i + u_{i-1} + h^2 \pi^2 \sin(\pi x_i).$$
Here, we utilized ```scipy.optimize``` module.
If we consider nonlinear equation, then rootfinding algorithm may be useful.
It is a kind of Newton's method, but the gradient is approximated by finite difference formula (secant method).