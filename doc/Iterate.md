# Iterate

`mv.Iterate` is the **single object every optimizer, linear solver, and diagonalizer in Maniverse operates on**. It:

1. Bundles the **objective** `Function` together with the list of **manifold** blocks and an optional list of **constraint** `Function`s.
2. Stores the current point and gradient as flattened ambient-coordinate vectors, while manifold operations interpret their tangent components.
3. Provides the *matrix-free* primitives (`Retract`, `TangentProjection`, `Inner`, `Hessian`, …) that the optimizers and linear solvers consume via the `initLinearSolverFor*` helpers.

## Construction

```python
M = mv.Iterate(objective, manifolds, cons_funcs)
# For an unconstrained problem, omit the third argument:
M = mv.Iterate(objective, manifolds)
```

| Argument | Type | Meaning |
|----------|------|---------|
| `objective`  | a `mv.Function` instance (subclass) | The principal objective function. |
| `manifolds`  | `list` of `mv.Manifold` instances | One entry per "block". Can be a mix of different concrete manifolds (see [Manifolds](Manifolds.md)). |
| `cons_funcs` | optional `list` of `mv.Function` instances | Scalar equality-constraint functions. Each function represents a constraint $g_i(x)=0$. Omit this argument for an unconstrained problem. |

The functions are stored **by reference** and must outlive the `Iterate`. 

## Frequently used properties and methods

### Point and layout

| Method / property | Returns | Meaning |
|-------------------|---------|---------|
| `M.getPoint()` | `list[np.ndarray]` | Current point, split into one array per manifold block. This is the preferred representation in user code. |
| `M.setPoint(ps, purify)` | — | Set the current point from one array per block. `purify=True` asks each manifold to correct its supplied point. |
| `M.Point` | `np.ndarray` | Current point flattened into one ambient-coordinate vector of length `M.TotalSize`. Usually read-only in user code. |
| `M.Manifolds` | `list[mv.Manifold]` | Manifold blocks owned by the iterate. `M.Manifolds[i].P` is the current point of block `i`. |
| `M.BlockParameters` | `list[tuple[int]]` | Internal block metadata. Each entry records a block's flattened offset, row count, and column count as `[offset, rows, columns]`. |
| `M.TotalSize` | `int` | Total number of ambient coordinates across all blocks. |
| `M.getDimension()` | `int` | Sum of the intrinsic tangent-space dimensions of all blocks. This can be smaller than `TotalSize`. |
| `M.getName()` | `str` | Names of the manifold blocks joined with `" * "`. |

### Evaluation and derivatives

| Method / property | Returns | Meaning |
|-------------------|---------|---------|
| `M.Calculate(P, derivatives)` | — | Evaluate the objective and constraints at the block list `P`. Flag `0` requests values and flag `1` requests Euclidean gradients; see [Function](Function.md). |
| `M.Value` | `float` | Combined augmented-Lagrangian value produced by `Calculate(P, [0])`: $f+\sum_i\lambda_i g_i+\frac{\rho}{2}\sum_i g_i^2$. |
| `M.setGradient()` | — | Project and combine the objective and constraint gradients into `M.Gradient`. Call `Calculate(P, [0, 1])` first so both $g_i$ and $\nabla g_i$ are current. |
| `M.Gradient` | `np.ndarray` | Flattened Riemannian gradient of the augmented Lagrangian. |
| `M.getGradient()` | `list[np.ndarray]` | The same augmented-Lagrangian gradient split into manifold blocks. |
| `M.setObjectiveGradient()` | — | Project the objective's current Euclidean gradient and store it in `M.ObjectiveGradient`, excluding constraints. Automatically called by `setGradient()`. |
| `M.ObjectiveGradient` | `np.ndarray` | Flattened bare objective gradient after `setObjectiveGradient()`. |
| `M.getObjectiveGradient()` | `list[np.ndarray]` | Bare objective gradient split into manifold blocks. |
| `M.ObjectiveHessian(X)` | `np.ndarray` | Riemannian objective Hessian applied to flattened direction `X`, excluding constraints. |
| `M.Hessian(X)` | `np.ndarray` | Augmented-Lagrangian Hessian applied to flattened direction `X`, including constraint terms. |

`M` does not expose `ObjectiveValue` or `ConstraintValues` properties. Read the bare values from the `Function` objects supplied to the constructor, as described below.

### Manifold operations

All vector arguments and return values in this table use flattened ambient-coordinate storage of length `M.TotalSize`.

| Method | Returns | Meaning |
|--------|---------|---------|
| `M.Inner(X, Y)` | `float` | Sum of the manifold inner products of directions `X` and `Y`. |
| `M.TangentProjection(X)` | `np.ndarray` | Project `X` onto the current tangent space. |
| `M.TangentPurification(X)` | `np.ndarray` | Apply each manifold's tangent-purification operation to `X`. |
| `M.Retract(S)` | `np.ndarray` | Retract tangent step `S` from the current point and return the resulting flattened point. It does not update `M`; use `setPoint` to store the result. |
| `M.InverseRetract(N)` | `np.ndarray` | Apply each manifold's inverse retraction between this iterate and another compatible `Iterate` `N`. |
| `M.TransportTangent(X, Y)` | `np.ndarray` | Transport `X` along tangent displacement `Y`, according to each manifold implementation. |
| `M.TransportManifold(X, N)` | `np.ndarray` | Transport `X` from this iterate to another compatible iterate `N`. |

### Constraints

`M.Constraints` contains one `mv.Constraint` wrapper for each function passed in `cons_funcs`, in the same order.
These wrappers hold the Riemannian derivatives and multiplier state used by the constrained optimizers.

| Method / property | Returns | Meaning |
|-------------------|---------|---------|
| `M.Constraints` | `list[mv.Constraint]` | Constraint wrappers corresponding to `cons_funcs`. An unconstrained iterate has an empty list. See below for `mv.Constraint` |
| `M.Rho` | `float` | Penalty parameter $\rho$ shared by all constraints. |
| `M.setLambda(values)` | — | Set one Lagrange multiplier per constraint. The list length must equal `len(M.Constraints)`. |
| `M.getLambda()` | `list[float]` | Return the multipliers in constraint order. |
| `M.calcLambda()` | `list[float]` | Estimate multipliers from the current objective and constraint gradients. Call the relevant evaluation and gradient methods first. |
| `M.OrthConsJacobian` | `np.ndarray`, shape `(M.TotalSize, len(M.Constraints))` | Matrix whose column `i` is the Gram–Schmidt-orthogonalized, flattened Riemannian gradient of constraint `i`. It is updated by `M.setGradient()` and used by `M.ConstraintProjection()`. |
| `M.ConstraintProjection(X)` | `np.ndarray` | Remove components of flattened direction `X` along the orthogonalized constraint gradients. Call `M.setGradient()` first. |

`mv.Constraint` is a wrapper of all data related to one specifc equality-constraint.

| Method / property | Returns | Meaning |
|-------------------|---------|---------|
| `constraint.Lambda` | `float` | Lagrange multiplier $\lambda_i$ for this constraint. Prefer `M.setLambda()` and `M.getLambda()` when handling all constraints together. |
| `constraint.Gradient` | `np.ndarray` | Flattened Riemannian gradient $\nabla g_i$ after `constraint.setGradient()`. |
| `constraint.setGradient()` | — | Project the wrapped function's current Euclidean gradient onto the tangent space and store it in `constraint.Gradient`. The constraint function must first be evaluated with derivative flag `1`. |
| `constraint.getGradient()` | `list[np.ndarray]` | Return `constraint.Gradient` split into manifold blocks. |
| `constraint.Hessian(X)` | `np.ndarray` | Apply the Riemannian Hessian of $g_i$ to flattened direction `X`. Required point-dependent Hessian data must already have been prepared by the constraint function. |
| `constraint.Manifolds` | `list[mv.Manifold]` | Internal manifold copies used to evaluate this constraint at the iterate's current point. |
| `constraint.TotalSize` | `int` | Total number of ambient coordinates across its manifold blocks. |
| `constraint.BlockParameters` | `list[tuple[int]]` | Internal block offsets and shapes used to split and assemble flattened vectors. |

The constraint gradients are usually prepared collectively:

```python
M.Calculate(M.getPoint(), [0, 1, 2])
M.setGradient()

first_constraint = M.Constraints[0]
print("lambda:", first_constraint.Lambda)
print("gradient blocks:", first_constraint.getGradient())
```

Calling `M.setGradient()` updates every constraint's `Gradient`,
the columns of `M.OrthConsJacobian`, and the augmented-Lagrangian gradient `M.Gradient`.

### Preconditioners

| Method / property | Returns | Meaning |
|-------------------|---------|---------|
| `M.Preconditioner` / `M.PreconditionerInv` | `callable` | A preconditioner and its inverse; both default to the identity. |
| `M.PreconditionerSqrt` / `M.PreconditionerInvSqrt` | `callable` | Square-root preconditioner pair; both default to the identity. |

The preconditioners for faster optimization usually utilizes the diagonal elements of the Hessian.
In Euclidean space, the diagonal elements are trivial;
For Riemannian manifolds, on the other hand, the diagonal elements
are defined by the local coordinates of the tangent space.

I am still trying to work out the best way
to expose the local coordinate chart to the users.
For now, users have to provide preconditioners
that can handle the local coordinates on their own.

See [Optimizers](Optimizers.md) and [LinearSolvers](LinearSolvers.md)
to see when preconditioners can be useful.

## Setting and accessing the current point

`M.setPoint(ps, purify)` takes two inputs:

1. `ps`: a list of NumPy arrays,
with exactly one array for each manifold block.
The arrays must appear in the same order and
have the same shapes as the points used to construct those manifolds.
Do not pass the flattened `M.Point` vector here.
2. `purify`: a Boolean. When true,
each manifold applies its point-purification operation,
such as restoring orthonormality for a Stiefel point;
When false, the arrays are accepted without that correction.

For a single block:

```python
manifold = mv.Stiefel(U0)
M = mv.Iterate(objective, [manifold])
M.setPoint([U], True)
```

For two blocks:

```python
U_manifold = mv.Stiefel(U0)
V_manifold = mv.Orthogonal(V0)
M = mv.Iterate(objective, [U_manifold, V_manifold])
M.setPoint([U, V], True)
```

Even a single-block problem requires a list:
use `[U]`, not `U`.
Calling `setPoint` updates both the flattened `M.Point` storage
and each `M.Manifolds[i].P`.

Use `M.getPoint()` to retrieve the current point as a list of arrays.
`M.Manifolds[i].P` and the other attributes listed in [Manifolds](Manifolds.md)
reflect the current block; `M.setPoint` keeps these in sync.

## Values and derivatives

The objective and constraint `Function` objects hold their bare values.
`M.Value`, `M.Gradient`, and `M.Hessian(X)` combine objective and
constraint contributions into the augmented Lagrangian.
Keep the supplied `Function` objects alive for the lifetime of `M`.

### Objective and constraint values

Evaluate with derivative flag `0` before reading values:

```python
objective = Obj()
constraint = Cons()
manifold = mv.Euclidean(initial_point)
M = mv.Iterate(objective, [manifold], [constraint])

M.Calculate(M.getPoint(), [0])
print("objective:", objective.Value)
print("constraint:", constraint.Value)
print("augmented value:", M.Value)
```

For multiple constraints, retain their objects in a list and read
`[constraint.Value for constraint in constraints]` in construction order.

Without constraints, `M.Value` equals `objective.Value`.
Otherwise, multipliers $\lambda_i$, and penalty $\rho=$ `M.Rho`,
`M.Value` equals the augmented-Lagrangian:

$$
L = f + \sum_i \lambda_i g_i + \frac{\rho}{2}\sum_i g_i^2.
$$

`objective.Value` remains the bare objective value $f$,
and each constraint function's `Value` remains its bare constraint value $g_i$.

### Gradients

Prepare values and Euclidean gradients, then assemble the Riemannian gradients:

```python
M.Calculate(M.getPoint(), [0, 1])
M.setGradient()
```

`M.setGradient()` updates:

- `M.ObjectiveGradient`: the flattened bare objective gradient.
- `M.Constraints[i].Gradient`: each flattened constraint gradient.
- `M.OrthConsJacobian[:, i]`: the sequentially Gram–Schmidt-orthogonalized
    gradient for constraint `i`, stored as column `i` of the matrix.
- `M.Gradient`: the flattened augmented-Lagrangian gradient.

With constraints:

Without constraints, `M.Gradient` and `M.ObjectiveGradient` are equal.
Otherwise, `M.Gradient` equals:

$$
\nabla L = \nabla f + \sum_i(\lambda_i+\rho g_i) \nabla g_i.
$$

`M.getGradient()` and `M.getObjectiveGradient()`
return the corresponding vectors split into manifold blocks.
These are lists of arrays.

#### Constraint projection

After `M.setGradient()`, `M.ConstraintProjection(X)` removes components
along the columns $q_i = \texttt{M.OrthConsJacobian}[:,i]$:

$$
X \mapsto X-\sum_i\frac{\langle X,q_i\rangle}{\langle q_i,q_i\rangle}q_i.
$$

`Lanczos(..., constraint=True)` uses this projection to remove the current
constraint-gradient directions from its search space.

### Hessian-vector products

Prepare point-dependent Hessian data with derivative flag `2` before
applying Hessians:

```python
M.Calculate(M.getPoint(), [0, 1, 2])
M.setGradient()
```

All Hessian methods below take a flattened direction `X`
and return a flattened vector:

- `M.ObjectiveHessian(X)`: bare objective Riemannian Hessian applied to `X`.
- `M.Constraints[i].Hessian(X)`: Riemannian Hessian of constraint $g_i$
    applied to `X`.
- `M.Hessian(X)`: augmented-Lagrangian Hessian applied to `X`.

Without constraints, `M.Hessian(X)` and `M.ObjectiveHessian(X)` are equal.
Otherwise, `M.Hessian(X)` is equal to:

$$
\nabla^2L[X] = \nabla^2 f[X] + \sum_i(\lambda_i+\rho g_i)\nabla^2g_i[X] + \rho\sum_i\langle\nabla g_i,X\rangle\nabla g_i.
$$

## Example: evaluating directional curvature

Assuming `Obj`, `ConsInner`, `U`, `V`, `np`, and `mv` are defined:

```python
U_man = mv.Stiefel(np.eye(10)[:, :6])
V_man = mv.Orthogonal(np.eye(3))
objective = Obj()
constraint = ConsInner()

M = mv.Iterate(objective, [U_man, V_man], [constraint])
# No need to setPoint because the initial values in U_man and V_man are not changed.
M.Calculate(M.getPoint(), [0, 1, 2])
M.setGradient()

w = M.TangentProjection(np.random.randn(M.TotalSize))
curv = M.Inner(w, M.Hessian(w))
```

`curv` is the augmented-Lagrangian curvature along `w`,
measured with the manifold metric.

## Pitfalls

* **Function lifetime**: the `Iterate` retains pointers to the objective and constraint function objects; keep those objects alive for at least as long as the `Iterate`.
* **Cached derivatives**: after changing the point, refresh the values and derivative data needed by subsequent gradient and Hessian calls.
* **Point representation**: `setPoint` expects a list of block arrays, not the flattened `M.Point`.
* **Ambient versus intrinsic dimension**: flattened points and directions have length `M.TotalSize`. `M.getDimension()` returns the intrinsic tangent-space dimension; do not use it as the flattened vector length.
* **`preconditioner` pairs**: if you set `PreconditionerSqrt` (used by `LBFGS` with a square-root scheme), you must also set `PreconditionerInvSqrt` to the corresponding inverse square root — they are both consumed in the two-loop recursion, and the product must round-trip (e.g. `Ainvsqrt*Asqrt` ≈ identity on `v`).
