# Nonlinear Solver

## Multi-Level Newton-Raphson

A quadratically convergent Newton-Raphson scheme has been proposed by [RabSanHsu:1979:mna](@citet).
Let us assume we have a block-nonlinear problem with unknowns $\hat{\bm{u}}$ and $\hat{\bm{q}}$ of the following form:

```math
\begin{aligned}
    \bm{\hat{f}_{\textrm{G}}}(\hat{\bm{u}}, \hat{\bm{q}}) &= 0 \\
    \bm{\hat{f}_{\textrm{L}}}(\hat{\bm{u}}, \hat{\bm{q}}) &= 0
\end{aligned}
```

where solving $\bm{\hat{f}_{\textrm{L}}}(\hat{\bm{u}}, \hat{\bm{q}}) = 0$ is easy to solve for fixed $\hat{\bm{u}}$.
If we can enforce this constraint, then we can rewrite the first equation by *implicit function theorem* as:
```math
\bm{\hat{f}_{\textrm{G}}}(\hat{\bm{u}}, \hat{\bm{q}}(\hat{\bm{u}})) = 0
```
Solving this modified problem with a Newton-Raphson algorithm requires a linearization around $\hat{\bm{u}}$, such that we have to solve at each Newton step the following linear problem
```math
\left( \frac{\partial  \bm{\hat{f}_{\textrm{G}}} }{\partial \hat{\bm{u}}} + \frac{\partial \bm{\hat{f}_{\textrm{G}}} }{\partial \hat{\bm{q}}} \frac{\mathrm{d} \hat{\bm{q}} }{\mathrm{d} \hat{\bm{u}}} \right) \Delta \hat{\bm{u}} = -\bm{\hat{f}_{\textrm{G}}}(\hat{\bm{u}}, \hat{\bm{q}}(\hat{\bm{u}}))
```
In the continuum mechanics community the system matrix is also known as the *consistent linearization*.

The last missing piece the implicit function part for the system matrix, which is determined by solving an additional linear system:
```math
\frac{\partial \bm{\hat{f}_{\textrm{L}}}(\hat{\bm{u}}^{i}, \hat{\bm{q}}^{i}) }{\partial \hat{\bm{q}}} \frac{\mathrm{d} \hat{\bm{q}} }{\mathrm{d} \hat{\bm{u}}} = -\frac{\partial \bm{\hat{f}_{\textrm{L}}}(\hat{\bm{u}}^{i}, \hat{\bm{q}}^{i}) }{\partial \hat{\bm{u}}}
```

### Using finite-element structure

Time discretization schemes applied to finite element semi-discretizations with $L_2$ variables (called *internal variables*) usually lead to block-nonlinear problems with *local-global structure*, as described above.
This is commonly found in continuum mechanics problems.
The local-global structure is simply a result of the algebraic decoupling of the internal variables, as they are associated with the quadrature points.
For the resulting nonlinear form of the space-time discretization with field unknowns $u$ and internal unknowns $q = (q_1, ... q_{nqp})$, we can write the finite element discretization formally as

```math
\begin{aligned}
    f_G(u,q,p,t) =& ... \\
    f_{Q}(u,q,p,t) =& ... \\
\end{aligned}
```

TODO picture with the fundamental decomposition operators

The internal unknowns $q$ are located at the quadrature points which implies the following structure

```math
\begin{aligned}
    f_{Q_1}(u,q_1,p,t) =& ... \\
    f_{Q_2}(u,q_2,p,t) =& ... \\
    \vdots \\
    f_{Q_{nqp}}(u,q_{nqp},p,t) = &... \\
\end{aligned}
```

### Example: Creep Test of 1D Linear Viscoelasticity

A simple linear viscoelastic material model in 1D in weak form is:

```math
\begin{aligned}
                0 =& \int_{\Omega} (E_0 \partial_x u(x,t) + E_1 (\partial_x u(x,t) + q(x,t))) \cdot \partial_x \delta u(x) \textrm{d}x + Neumann \\
\partial_t q(x,t) =& \frac{E_1}{\eta_1} (\partial_x u(x,t)-q(x,t))
\end{aligned}
```

Assuming we have a single 1D element $\Omega = [-1,1]$ with linear ansatz functions and Gauss-Legendre quadrature (i.e. 2 points) the Neumann conditon $\partial_x u(1,t) = 1$ and the Dirichlet condition $u(-1,t) = 0$, then applying a Galerkin semi-discretization yields the following linear DAE in mass matrix form:

```math
\begin{aligned}
            0   &=  \tilde{u}_1 \\
            0   &=  0.5\left(-(E_0 + E_1) \tilde{u}_1 + (E_0 + E_1) \tilde{u}_2 - E_1 \tilde{q}_1 - E_1 \tilde{q}_2 + 1\right) \\
d_t \tilde{q}_1 &= \frac{E_1}{\eta_1} (-\tilde{u}_1+\tilde{u}_2-\tilde{q}_1) \\
d_t \tilde{q}_2 &= \frac{E_1}{\eta_1} (-\tilde{u}_1+\tilde{u}_2-\tilde{q}_2)
\end{aligned}
```
or, after condensing the first equation,
```math
\begin{aligned}
            0   &= 0.5\left((E_0 + E_1) \tilde{u}_2(t) - E_1 \tilde{q}_1(t) - E_1 \tilde{q}_2(t) + 1\right) \\
d_t \tilde{q}_1(t) &= \frac{E_1}{\eta_1} (-\tilde{u}_1+\tilde{u}_2(t)-\tilde{q}_1(t)) \\
d_t \tilde{q}_2(t) &= \frac{E_1}{\eta_1} (-\tilde{u}_1+\tilde{u}_2(t)-\tilde{q}_2(t))
\end{aligned}
```

Applying the Backward Euler in time to this system we obtain the ,,nonlinear'' system
```math
\begin{aligned}
 0 &= 0.5\left((E_0 + E_1) \hat{\tilde{u}}^n_2 - E_1 \hat{\tilde{q}}_1 - E_1 \hat{\tilde{q}}_2 + f(t_n)\right) &= f_G(\hat{\tilde{u}}_2, \hat{\tilde{q}}_1, \hat{\tilde{q}}_2) \\
 0 &= \hat{\tilde{q}}^n_1 - \hat{\tilde{q}}^{n-1}_1 - \Delta t_n \frac{E_1}{\eta_1} (-\hat{\tilde{u}}_1+\hat{\tilde{u}}_2-\hat{\tilde{q}}_1) &= f_{Q_1}(\hat{\tilde{u}}_2, \hat{\tilde{q}}_1) \\
 0 &= \hat{\tilde{q}}^n_2 - \hat{\tilde{q}}^{n-1}_2 - \Delta t_n \frac{E_1}{\eta_1} (-\hat{\tilde{u}}_1+\hat{\tilde{u}}_2-\hat{\tilde{q}}_2) &= f_{Q_2}(\hat{\tilde{u}}_2, \hat{\tilde{q}}_2) \\
\end{aligned}
```

where we can observe that the internal problems are decoupled. The system can now be solved with the multi-level Newton-Raphson as described above.
