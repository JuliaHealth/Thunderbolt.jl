# Time Integration

## [Homotopy path methods](@id theory_homotopy-path-methods)

Homotopy path methods solve nonlinear problems with pseudo-time $t$ on some time interval $[t_0, t_1]$.
An initial guess is provided for the first nonlinear solve.
Formally we can write down the problem as follows.
Find $u(t)$ such that

```math
0 = F(u(t), p, t) \qquad \text{on} \; [t_0, t_1],
```

where $u$ usually descibes the displacement of some mechanical system and the
operator $F$ contains some mechanical load, hence a subclass of these methods are so-called *load stepping techniques*.
For mechanical problems we obtain systems with this form if we assume that inertial terms are neglibile ($||\rho d^2_tu|| \approx 0$).

## [Operator Splitting](@id theory_operator-splitting)

For operator splitting procedures we assume that we have some time-dependent
problem with initial condition $u_0 := u(t_0)$ and an operator $F$ describing
the right hand side. We assume that $F$ can be additively split into $N$
suboperators $F_i$. This can be formally written as

```math
d_t u(t) = F(u(t), p, t) = F_1(u(t), p, t) + ... + F_N(u(t), p, t) \, .
```

We call $t$ time the $u(t)$ the *state* of the system. This way we can
define subproblems

```math
\begin{aligned}
    d_t u(t) &= F_1(u(t), p, t) \\
             & \vdots \\
    d_t u(t) &= F_N(u(t), p, t)
\end{aligned}
```

Now, the key idea of operator splitting methods is that solving the subproblems
can be easier, and hopefully more efficient, than solving the full problem.
Arguably the easiest algorithm to advance the solution from $t_0$ to some time
point $t_1 > t_0$ is the Lie-Trotter-Godunov operator splitting [Lie:1880:tti,Tro:1959:psg,God:1959:dmn](@cite).
Here the subproblems are solved consecutively, where the solution of one
subproblem is taken as the initial guess for the next subproblem, until we have
 solved all subproblems. In this case we have constructed an _approximation_
 for $u(t_1)$.

More formally we can write the Lie-Trotter-Godunov scheme [Lie:1880:tti,Tro:1959:psg,God:1959:dmn](@cite) as follows:

```math
\begin{aligned}
    \text{Solve} \quad d_t u^1(t) &= F_1(u^1(t), p, t) & & \quad \text{on} \; [t_0, t_1] \; \text{with} \; u^1(t_0) = u_0 \\
    \text{Solve} \quad d_t u^2(t) &= F_2(u^2(t), p, t) & & \quad \text{on} \; [t_0, t_1] \; \text{with} \; u^2(t_0) = u^1(t_1) \\
             & \vdots & & \\
    \text{Solve} \quad d_t u^N(t) &= F_N(u^N(t), p, t) & & \quad \text{on} \; [t_0, t_1] \; \text{with} \; u^N(t_0) = u^{N-1}(t_1)
\end{aligned}
```
Such that we obtain the approximation $u(t_1) \approx u^{N-1}(t_1)$. The
approximation is first order in time, as we will show in the next section.

Probably the most widely spread application for operator splitting schemes is
the solution forreaction diffusion systems. These have the form

```math
d_t u(t) = Lu + R(u)
```

where $L$ is some linear operator, usually coming from the linaerization of
diffusion opeartors and a nonlinear reaction part $R$ which has some interesting
locality properties. This locallity property usually tells us that the time
evolution of $R$ natually decouples into many small blocks. This way we only
have to solve for the time evolution of a linear problem $d_t u(t) = Lu$ and a
set of many very small nonlinear problems $d_t u(t) = R(u)$.

### Analysis of Lie-Trotter-Godunov

It should be noted that even if we solve all subproblems analytically, then
operator splitting schemes themselves almost always come with their own
approximation error, which is simply called the splitting error. For linear
problems this error can vanish if all suboperators $F_i$ commute, i.e. if
$F_j \cdot F_i = F_i \cdot F_j$ for all $1 \leq i,j \leq N$, which can be shown
with the Baker-Campbell-Hausdorff formula. Let us investigate the convergence
order for two bounded linear operators $L_1$ and $L_2$, i.e. on the following
system of ODEs

```math
d_t u = L_1 u + L_2 u \, .
```

Here the exact solution $u$ at time point $t$ for some initial condition at $t_0 = 0$ is

```math
u(t) = e^{(L_1 + L_2)t} u_0 \, ,
```

while the solution for the Lie-Trotter-Godunov scheme is

```math
\tilde{u}(t) = e^{L_1t}e^{L_2t} u_0 \, .
```

The local truncation error can be written as

```math
\epsilon(t) = ||e^{L_1t}e^{L_2t} - e^{(L_1 + L_2)t}|| \, ||u_0||
```

if we now replace the exponentials with their definitions we obtain for the first norm

```math
\begin{aligned}
&||(I + tL_1 + \frac{h^2}{2}L_1^2 + ...)(I + tL_2 + \frac{h^2}{2}L_2^2 + ...) - (I + t(L_1 + L_2) + \frac{h^2}{2}(L_1+L_2)^2 + ...)||\\
=& ||\frac{h^2}{2} (L_1 L_2 - L_2 L_1) + ... || \leq \frac{h^2}{2} || (L_1 L_2 - L_2 L_1) || + O(h^3)
\end{aligned}
```

This shows that the local truncation error is O(h^2) and hence the scheme is first order accurate.

Showing stability is also straight forward. We assumed that $L_1$ and $L_2$ are
bounded, so we obtain for all time points $t' < t$ and all repeated subdivisions
$n \in \mathbb{N}$ the following bound

```math
||(e^{L_1\frac{t'}{n}}e^{L_2\frac{t'}{n}})^n||
\leq ||e^{L_1\frac{t'}{n}}e^{L_2\frac{t'}{n}}||^n
\leq ||e^{L_1\frac{t'}{n}}||^n ||e^{L_2\frac{t'}{n}}||^n
\leq e^{||L_1||t'} e^{||L_2||t'}
\leq e^{||L_1||t} e^{||L_2||t}
\leq C < \infty
```

which implies stability of the scheme.

## [Newmark-$\beta$ for second order systems](@id theory_newmark)

Dropping the assumption that inertial terms are negligible, which the homotopy path methods above
rest on, leaves the balance of momentum in its second order form. Semi-discretized in space it reads

```math
M \, d^2_t u + f^{\mathrm{int}}(u) = f^{\mathrm{ext}}(t) \, ,
```

with the mass matrix $M_{ij} = \int_\Omega \rho \, N_i \cdot N_j \, \mathrm{d}\Omega$. Note that
$f^{\mathrm{int}}$ is in general nonlinear in $u$ -- this is where the presentation below departs
from the linear elasticity textbook case.

The Newmark-$\beta$ family [New:1959:amc](@cite) advances the triple $(u_n, v_n, a_n)$ of
displacement, velocity and acceleration by

```math
\begin{aligned}
u_{n+1} &= u_n + \Delta t \, v_n + \Delta t^2 \left[ \left( \tfrac{1}{2} - \beta \right) a_n + \beta \, a_{n+1} \right] \\
v_{n+1} &= v_n + \Delta t \left[ (1 - \gamma) a_n + \gamma \, a_{n+1} \right] \, .
\end{aligned}
```

The parameter $\gamma$ controls dissipation and $\beta$ controls stability. The choice
$\beta = 1/4, \gamma = 1/2$ -- the *average acceleration* rule -- is unconditionally stable, second
order accurate and conserves energy. Any $\gamma > 1/2$ introduces numerical damping and reduces the
scheme to first order.

### Displacement form

It is convenient to split off the parts of the update that are known before the step,

```math
\tilde{u} = u_n + \Delta t \, v_n + \left( \tfrac{1}{2} - \beta \right) \Delta t^2 a_n \, , \qquad
\tilde{v} = v_n + (1 - \gamma) \Delta t \, a_n \, ,
```

so that the update formulas become $u_{n+1} = \tilde{u} + \beta \Delta t^2 a_{n+1}$ and
$v_{n+1} = \tilde{v} + \gamma \Delta t \, a_{n+1}$.

Textbook presentations now solve for the acceleration, inserting these into the balance of momentum
to obtain a linear system with the *effective mass matrix*
$M_{\mathrm{eff}} = M + \beta \Delta t^2 K$. Its appeal is that $M_{\mathrm{eff}}$ is constant and can
be factorized once outside the time loop -- but that rests on $K$ being constant, which holds for
linear elasticity and for none of the material models in this package.

We therefore solve for the **displacement**, which is the quantity the constitutive model is a
function of. Inverting the first update formula expresses the acceleration in terms of the unknown,

```math
a(u) = \frac{u - \tilde{u}}{\beta \Delta t^2} \, ,
```

and the nonlinear problem of one step is

```math
r(u) = M \, a(u) + f^{\mathrm{int}}(u) - f^{\mathrm{ext}}(t_{n+1}) = 0 \, , \qquad
\frac{\partial r}{\partial u} = K(u) + \frac{1}{\beta \Delta t^2} M \, .
```

For a linear material this is the acceleration form scaled by $\beta \Delta t^2$, so nothing is lost.
Once $u_{n+1}$ is found, $a_{n+1}$ and $v_{n+1}$ follow from the formulas above.

### What a rate dependent material sees

Materials whose internal variables follow $d_t Q = L(F, d_t F, Q)$ -- the sarcomere models are the
motivating case -- need the deformation rate, and its linearization with respect to the unknown. Under
Newmark the velocity is *not* the backward difference $(u - u_n)/\Delta t$. Substituting $a(u)$ into
the second update formula gives

```math
v(u) = \tilde{v} + \frac{\gamma}{\beta \Delta t} \left( u - \tilde{u} \right) \, ,
```

which is affine in the unknown. Writing any affine function as a slope times a displacement from its
root,

```math
v(u) = \frac{\partial v}{\partial u} \left( u - u_v \right) \, , \qquad
\frac{\partial v}{\partial u} = \frac{\gamma}{\beta \Delta t} \, , \qquad
u_v = \tilde{u} - \tilde{v} \left/ \frac{\partial v}{\partial u} \right. \, ,
```

gives the two quantities an element needs in order to form $\dot{F} = \mathrm{grad}(v)$ and to
contribute $\partial P / \partial \dot{F} \cdot \partial \dot{F} / \partial u$ to the tangent. Backward
Euler is the same statement with $\partial v / \partial u = 1/\Delta t$ and $u_v = u_n$, which is why
that scheme never needed the concept: its root *is* the previous solution.

!!! note "Two time quantities, not one"
    A scheme hands the element the reconstruction above **and**, separately, the timestep $\Delta t$
    that the internal variable integrates over -- the local problem $(Q - Q_n)/\Delta t = L(F, \dot{F}, Q)$
    is first order in $Q$ regardless of what the global scheme does with $u$. Under backward Euler the
    slope happens to equal $1/\Delta t$ and the two collapse into one number; under Newmark they differ
    by $\gamma/\beta$. Conflating them silently linearizes a different problem than the residual poses.

## References

```@bibliography
Pages = ["topics/time-integration.md"]
Canonical = false
```
