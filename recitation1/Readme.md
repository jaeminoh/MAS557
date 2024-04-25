# Recitation 1

## Physics-informed Neural Networks (PINNs)

## 1d Boundary Layer Problem
code: `pinn_singular1d.py`

Domain: $(0, 1)$.
Differential equation and boundary condition:
$$-\epsilon u_{xx} - u_x = 1, \quad u(0) = u(1) = 0.$$

An analytic solution is available:
$$u(x) = -x + \frac{1 - e^{-x / \epsilon}}{1 - e^{-1 / \epsilon}}.$$

PINN:
$$u^\theta(x) = x (1-x) \text{MLP}^\theta(x),$$
to impose the boundary condition strictly.


## 2d Poisson Equation on an L-shaped Region
code: `pinn_poisson2d.py`

Domain: $(-1, 1)^2 \setminus (0, 1)^2$.
Differential equation:
$$-\Delta u = 1.$$
Zero Dirichlet boundary condition.

Monte Carlo integration with Sobol sequences.


# (Physics-informed) Deep Operator Networks (DeepONets)

## Parametrized 1d Poisson Equation
code: `pideeponet_poisson1d.py`

Domain: $(0, 1)$.
Source term: $f(x;a) = -a \pi^2 \sin(\pi x)$, for $a \in (-1, 1)$.
Differential Equations and boundary condition:
$$u_{xx}(x; a) = f(x;a), \quad u(0;a) = u(1;a) = 0.$$

We need a branch network ($b^\theta$) and a trunk network ($t^\theta$) to construct a DeepONet.
We multiply $x(1-x)$ by the output of the trunk network to strictly impose the zero Dirichlet boundary condition.
The formula for DeepONet is:
$$O^\theta(x;a) = b^\theta(a) \cdot t^\theta(x),$$
where $\cdot$ is the standard inner product.

Loss:
$$\frac{1}{N_a}\sum_{i=1}^{N_a} \int_0^1 \left(O^\theta_{xx}(x;a_i) - f(x;a_i)\right)^2 dx.$$


# Deep Ritz Method

## 1d Poisson Equation

Domain: $(0, 1)$.
Differential Equations and boundary condition:
$$u_{xx}(x) = -\pi^2 \sin(\pi x) = f(x), \quad u(0) = u(1) = 0.$$

Weak form:
$$\int -u_{xx} v dx = \int -f v dx.$
Integration by parts (test function $v$ must satisfy $v(0) = v(1) = 0$:
$$\int u_x v_x dx = \int -f v dx.$$
Define
$$a(u,v) = \int u_x v_x dx, \quad L_f(v) = \int -f v dx.$$
Define
$$E(u) = \frac{1}{2}a(u, u) - L_f(u).$$
It is known that the minimizer of $E(u)$, say $u^\star$ satisfies
$$a(u^\star, v) = L_f(v),$$
for all suitable test functions $v$.

So our loss function is
$L(\theta) = E(u_\theta)$.
After minimization w.r.t. $\theta$, we expect $u^\theta \approx u^\star$.

However, it seems the quality of the integral affects the quality of the solution.

### Monte Carlo Integration, Uniform Distribution
code: `deepritz.py`

### Trapezoidal Rule
code: `deepritz_trapz.py`

### Gauss-Legendre Quadrature
code: `deepritz_gauss.py`