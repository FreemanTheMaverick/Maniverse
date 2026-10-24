# Optimizers

The optimizers take an `mv.Iterate` (see [Iterate](Iterate.md)),
which holds the objective, manifold point, and optional equality constraints.
The Python functions return `True` when their convergence test succeeds
and `False` when their outer iteration limit is reached.
`AugmentedLagrangian` also raises `RuntimeError` if an inner optimizer returns `False`.

Common arguments:

| Argument | Type | Meaning |
|----------|------|---------|
| `M` | `mv.Iterate` | The iterate to update. |
| `tol` | `sequence[float]` (length 3) | Three method-specific tolerances. Newton and L-BFGS use target change, gradient norm, and step norm; Anderson uses target change, residual norm, and step norm. |
| `max_iter` | `int` | Maximum number of outer iterations. |
| `output` | `int` | Console-output control (`0` off, positive on). |

## Newton's method

Trust-region Riemannian Newton with a matrix-free linear solver for each Hessian subproblem.

```python
mv.Newton(M, tr, ls, tol, max_iter, output)
```

`tr` is an `mv.TrustRegion`;
`ls` is an initialized `mv.LinearSolver` wired to this same `M` with `mv.initLinearSolverForNewton(ls, M)`.
See below and [Linear Solvers](LinearSolvers.md).

It is a second-order method that uses Hessian information to find a stationary point.
The trust region limits steps when the local quadratic model may be unreliable.

### Overview

At the initial point Newton evaluates the objective and derivatives.
On subsequent iterations it finds a step in the current trust region
by solving the quadratic model with `ls`,
retracts the step, and compares actual and predicted objective changes.
It accepts a trial if the ratio exceeds `tr.RhoThreshold`
or the step norm is below the third tolerance.
A rejected trial shrinks the radius and is retried.
The predicted change is `⟨grad, S⟩ + 0.5⟨S, Hessian(S)⟩`;
for a successful descent step,
both predicted and actual changes are normally negative,
giving a positive ratio.

Newton converges immediately if the initial gradient norm is below `tol[1]`.
On later iterations it converges if either:

* the gradient norm is below `tol[1]`, the absolute target change is below `tol[0]`, and the step norm is below `tol[2]`; or
* the gradient norm is below `tol[1] ** 2` or the step norm is below `tol[2] ** 2`.

The second condition is an implementation-specific fallback.

### Choosing the linear solver

`Maniverse` has two built-in linear solver,
conjugate gradient (`ConjugateGradient`) and minimal residual (`MinRes`).
Choose based on the goal and on curvature over the optimization path,
not just the Hessian at the initial point:

* **Find a local minimum; Hessian is positive definite throughout:**
This is the natural case for CG and is a good default for a well-conditioned problem,
although either is valid.
* **Find a local minimum; Hessian may be indefinite at some iterates:**
Either is valid with `FrownNPC=True`; CG is theoretically slightly better.
This flag tells solvers to detect non-positive curvature in the trust-region subproblem
and can return a boundary step.
An indefinite Hessian at an intermediate point does not rule out convergence to a minimum,
but Newton does not guarantee which stationary point it reaches.
Check the curvature at the final point if minimum classification matters.
* **Search for a saddle point with a non-positive-definite Hessian:**
use `mv.MinRes` with `FrownNPC=False`,
so the inner solve does not stop when it encounters non-positive curvature.
This can let Newton follow indefinite Newton directions toward a saddle,
but it is not very efficient and does not guarantee convergence to the desired saddle.
Saddle-point search theory and practical guidance are less general than for minimization,
so convergence failure is possible;
the method may also be sensitive to initialization or reach a different stationary point.
Newton's stopping test checks stationarity, not whether the result is a minimum or saddle.
Check the curvature at the final point if minimum classification matters.

Both inner solvers are matrix-free and
use the Hessian-vector callback installed by `initLinearSolverForNewton`.
If an inner solve converges slowly,
consider setting `M.Preconditioner` and
adjusting its residual and quadratic-model tolerances or iteration limit.
See [Linear Solvers](LinearSolvers.md) for constructor details and trade-offs.

Before passing a solver to `Newton`, wire it to the same iterate with
`mv.initLinearSolverForNewton(ls, M)`.
This installs the callbacks on the solver as follows:

| Solver member | Newton operation |
|---------------|------------------|
| `ls.dot(X, Y)` | `M.Inner(X, Y)` |
| `ls.proj(X)` | `M.TangentProjection(X)` |
| `ls.A(X)` | `M.Hessian(X)` |
| `ls.P(X)` | `M.Preconditioner(X)` |

The helper does not set `ls.b`:
Newton updates it to `-M.Gradient` before each inner solve,
then calls `ls.Calculate(R)` and obtains the step with `ls.Find(R)`.
These callbacks refer to `M`, so use the same `Iterate` for wiring and optimization.

### Trust Region

The trust-region radius limits each Newton step to a neighborhood
where the local quadratic model is expected to be useful.
Newton compares the model's predicted change with the actual change
after the trial step, accepting or rejecting the step and adjusting the radius
according to how well they agree.

```python
tr = mv.TrustRegion()  # R0=1, RhoThreshold=0.1
tr.R0 = 1.0
tr.RhoThreshold = 0.1
tr.Update = lambda R, rho, Snorm: ...
```

The default radius update is

$$
R_{\mathrm{next}} =
\begin{cases}
\min(0.25R,\,0.75\lVert S\rVert), & \rho < 0.25, \\
\min(2R,\,R_0), & \rho > 0.75\ \text{or}\ \left|\lVert S\rVert^2-R^2\right| < 10^{-10}, \\
R, & \text{otherwise}.
\end{cases}
$$

Here `R` is the current radius and `R0` is its initial value; the second condition detects a step on the trust-region boundary. The custom `Update` callable receives `(R, rho, Snorm)`.

### Setup and example

Create the `Iterate`, solver, and trust-region object before calling Newton.
`initLinearSolverForNewton` installs `dot`, tangent projection,
Hessian application, and preconditioner callbacks on the solver.
`M.Preconditioner` defaults to the identity and can be replaced before optimization.

```python
tr = mv.TrustRegion()
cg = mv.ConjugateGradient(1, [1e-4, 1e-4], M.getDimension(), 1)
mv.initLinearSolverForNewton(cg, M)

converged = mv.Newton(M, tr, cg, [1e-5, 1e-5, 1e-5], 20, 1)
```

## L-BFGS

Limited-memory Riemannian BFGS with Armijo backtracking.

```python
mv.LBFGS(M, tol, max_mem, max_iter, c1, tau, ls_max_iter, output)
```

L-BFGS is useful for smooth problems
where forming or solving with a full Hessian is impractical;
it approximates curvature from recent gradient information instead.

### Overview

After obtaining the Riemannian gradient,
L-BFGS uses recent transported step and gradient-change pairs
to build a limited-memory inverse-Hessian direction
with its two-loop recursion.
At most, `max_mem` latest tangent-vector pairs are stored.
The oldest pair is discarded when the history exceeds `max_mem`.
After each update, it refreshes the gradient and transports the history
to the new point before constructing the next direction.
It does not form or solve a Hessian system.

Convergence requires the gradient norm below `tol[1]`;
after the first iteration,
it additionally requires both the absolute target change below `tol[0]`
and the step norm below `tol[2]`.
If a later line search fails,
L-BFGS clears its history and retries with steepest descent;
if that search also fails, it stops and returns `False`.

Stored tangent vectors are transported as the point changes.
Optional square-root preconditioning uses
`M.PreconditionerSqrt` and `M.PreconditionerInvSqrt` in the two-loop recursion.
Set these to mutually inverse square-root operators for the same SPD preconditioner;
by default, the preconditioners are identity.

### Armijo Backtracking line search

```python
success = mv.ArmijoBacktracking(M, S, c1, tau, max_iter, output)
```

For `LBFGS`, `c1` controls sufficient decrease and `tau` is the step shrink factor;
common choices are `c1=1e-4` and `0 < tau < 1` (for example, `0.5`).
`ls_max_iter` limits backtracking attempts: the first outer iteration tests the full step once,
and later searches use up to `ls_max_iter` attempts.

`S` is a tangent-space search direction before line-search scaling, usually a descent direction such as `-M.Gradient`. The routine tries `alpha=1`, shrinking by `tau` each attempt, until the retracted point satisfies

$$
L(\operatorname{Retract}(\alpha S))
\le L + c_1\alpha\langle S, \operatorname{grad} L\rangle_M.
$$

Here `L` and `grad L` are evaluated at the current iterate, and the inner product is `M.Inner`.

On success it scales `S` in place by `alpha` and returns `True`; otherwise it returns `False`. It checks at most `max_iter` candidate steps. A caller using it directly remains responsible for applying the accepted step to the iterate.

## Anderson acceleration

Anderson mixing is intended for fixed-point updates.

```python
mv.Anderson(M, tol, beta, max_mem, max_iter, output)
```

The implementation uses `M.Gradient` as its residual,
so the objective's gradient sign and scaling must match the fixed-point update you intend.
`beta` scales the residual step (often between `0.1` and `0.5`);
`max_mem` limits the stored history.
Its convergence test uses residual norm `tol[1]`,
plus target change `tol[0]` and step norm `tol[2]` after the first iteration.
For ordinary smooth minimization,
prefer Newton or L-BFGS unless the problem naturally has a fixed-point form.

For now, `Anderson` is less tested,
because I have not discovered any scenarios where Anderson acceleration is necessary.
If you plan to use Riemannian Anderson acceleration for your own project,
feel free to contact me for discussion.

## Augmented Lagrangian method

The augmented Lagrangian method handles equality-constrained optimization
by repeatedly solving an inner problem and updating the constraint multipliers and penalty.
This drives constraint violations toward their requested tolerances
while allowing the inner problem to be handled by an existing optimizer.

The wrapper applies this outer iteration to constraints $g_i(P) = 0$.
Register the constraints when constructing `M` (see [Iterate: Constraints](Iterate.md#constraints)).
`tol` must contain one absolute violation threshold for each registered constraint,
in the same order as `M.Constraints`.

```python
mv.AugmentedLagrangian(init_rho, theta_rho, theta_sigma, cons_tol, max_out_iter, output)(inner_solver)
```

| Argument | Type | Meaning |
|----------|------|---------|
| `init_rho` | `float` | Initial penalty parameter $\rho$; use a positive value (often `1.0`). |
| `theta_rho` | `float` | Factor multiplying $\rho$ when constraint violation has not decreased enough; choose greater than `1` (for example, `3.3`). |
| `theta_sigma` | `float` | Required fraction of the previous maximum violation to meet before the penalty stays unchanged; typically between `0` and `1` (for example, `0.8`). |
| `cons_tol` | `sequence[float]` (length = number of constraints) | Absolute tolerance for each constraint value. |
| `max_out_iter` | `int` | Maximum number of outer penalty iterations. |

Before the first outer iteration, ALM estimates the multipliers from the current iterate
and sets $\rho$ to `init_rho`.
During each inner solve, the current multipliers $\lambda_i$ and penalty $\rho$ are held fixed
while `inner_solver` optimizes the current augmented Lagrangian.
After the inner solve, ALM checks each constraint against its corresponding `cons_tol`.
If all constraints meet tolerance, it returns `True`.
Otherwise, the outer loop updates the multipliers as
$\lambda_i \leftarrow \lambda_i + \rho g_i(P)$.
From the second outer iteration onward, it also multiplies $\rho$ by `theta_rho`
if the maximum absolute constraint violation is greater than
`theta_sigma` times its value after the preceding inner solve.
The loop stops with `False` if the constraints remain outside tolerance after `max_out_iter` iterations.

```python
solver = mv.AugmentedLagrangian(1.0, 3.3, 0.8, [1e-5], 25, 0)(mv.Newton)

converged = solver(M, tr, cg, [1e-5, 1e-5, 1e-5], 12, 0) # The same arguments with the unconstrained solver
```

The returned wrapper passes its arguments through to the inner optimizer.
If that optimizer returns `False`, the wrapper raises `RuntimeError`.
Use `theta_rho > 1` and typically `0 < theta_sigma < 1`.

The inner solver can be the built-in `Newton` and `LBFGS` or
any user-defined callable with the `M` as the first argument.