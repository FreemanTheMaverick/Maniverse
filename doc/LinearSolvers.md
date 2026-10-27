# Linear Solvers

`Maniverse` implements a `LinearSolver` class derived by `ConjugateGradient` and `MinRes` (minimal residual).
They apply the matrix to solve through a callback rather than forming a dense matrix.
They can be used independently, or more frequently,
as the matrix-free subproblem sover used by `mv.Newton`.
In the latter case, you need to wire it to the `Iterate`,
and pass it to Newton; see [Optimizers](Optimizers.md).

## Overview

A `LinearSolver` instance is constructed with

```python
mv.LinearSolver(FrownNPC, [tol_quad, tol_res], MaxIter, Verbose)
```

| Argument | Type | Meaning |
|----------|------|---------|
| `FrownNPC` | `bool` | Whether to stop the Krylov solve when it detects non-positive curvature. |
| `[tol_quad, tol_res]` | pair of `float` | Relative quadratic-model change and relative linear residual tolerances. The solve stops when either tolerance is met. |
| `MaxIter` | `int` | Maximum number of inner Krylov iterations. This is an iteration limit, not a dimension argument; examples often use `M.getDimension()` as a convenient cap. |
| `Verbose` | `bool` | Whether to print inner-solver progress. |

The base class exposes these callbacks and methods:

| Member | Purpose |
|--------|---------|
| `dot(X, Y)` | Inner product used for Krylov norms and trust-region lengths. |
| `proj(X)` | Projects `X` into the subspace allowed by the subproblem. |
| `A(X)` | Applies the linear operator without assembling a matrix. |
| `P(X)` | Applies the preconditioner used by the iterative method. |
| `b` | Right-hand side of the linear system. |
| `Calculate(R)` | Runs the iterative solve for trust-region radius `R` and stores its candidate steps. |
| `Find(R)` | Returns the computed step for radius `R`. |

`X` and `Y` are both flattened vectors instead of lists of matrices.
`R` can be set as a very large number for non-trust-region solve.

```python
# A is symmetric positive definite; b is a one-dimensional NumPy array.
ls = mv.ConjugateGradient(True, [1e-8, 1e-8], len(b), False)
ls.dot = lambda x, y: float(x @ y)
ls.proj = lambda x: x
ls.A = lambda x: A @ x
ls.P = lambda x: x
ls.b = b

R = 1e6  # Choose the radius for the trust-region subproblem.
ls.Calculate(R)
step = ls.Find(R)
```

## Steihaug-Toint procedure

Trust-region steps must remain inside the ball of radius `R`.
When an update along a search direction would cross its boundary,
the Steihaug-Toint procedure finds the step length
that truncates the update at the boundary.

```python
mv.SteihaugToint(dot, v, p, R)
```

Returns the interpolate factor `t` that
puts `v + t * p` on the trust-region boundary of radius `R`,
measured using `dot`.
CG and MinRes, although in different ways, use it internally
when their candidate step reaches the boundary
or negative curvature is detected (if `FrownNPC=True`).
Most users do not need to call it directly.

## Built-in linear solvers

### Conjugate gradient

Conjugate Gradient builds conjugate directions
to solve the symmetric positive-definite system $A(S)=b$.
It is typically efficient when `A` is reasonably conditioned.

If the trust-region boundary is reached
or negative curvature is detected (if `FrownNPC=True`),
CG arrives at the interpolated solution on the boundary
via the Steihaug-Toint procedure.

With `FrownNPC=False`,
CG behaviours are undefined if `A` is indefinite.

### Minimal residual

MinRes reduces the linear residual in a Krylov subspace
and is generally preferred
when `A` may be symmetric indefinite or poorly conditioned.

MR adopts the interpolated solution like CG when
(1) the trust-region boundary is reached by the **first** trial vector in the sequence
(i.e., the gradient descent direction) or
(2) a negative curvature is detected (if `FrownNPC=True`).
Otherwise, when the trust-region boundary is reached by a following trial vector
without a detected negative curvature,
it falls back to the last candidate within the trust region as the solution.
This is because the paths between neighbouring MR trial vectors are generally
**not** monotonically decreasing for the quadratic subproblem nor the residual,
and interpolation on them tends to result in large errors.

With `FrownNPC=False`, it continues through negative curvature;
this can be tried for saddle-point search with Newton,
but may be inefficient and does not guarantee convergence to a desired saddle.
Newton's outer loop still decides whether to accept the resulting step.
