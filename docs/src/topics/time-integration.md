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

## References

```@bibliography
Pages = ["topics/time-integration.md"]
Canonical = false
```
