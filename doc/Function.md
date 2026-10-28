# Functions

Every objective and equality constraint in Maniverse is represented by a subclass of `Function`.

The exact C++ interface is:

```cpp
class Function{
public:
    double Value = 0;
    std::vector<Eigen::MatrixXd> Gradient;

    virtual void Calculate(
        std::vector<Eigen::MatrixXd> P,
        std::vector<int> derivatives
    );

    virtual std::vector<Eigen::MatrixXd> Hessian(
        std::vector<Eigen::MatrixXd> X
    ) const;
};
```

The base methods are unimplemented.
Override the operations required by your solver.

## Data members

### `Value`

Objective or constraint value at the current point.

```python
self.Value = 1.23
```

### `Gradient`

Euclidean gradient, stored as one matrix per manifold block.

```python
self.Gradient = [G0, G1, ...]
```

The number, order, and shapes of blocks must match the manifold blocks passed to `mv.Iterate`.

### Additional attributes

You may also add arbitrary attributes to cache intermediate quantities for reuse between value, gradient, and second-order computations.

## Methods

Subclasses should override the operations required by the solver being used.
`Calculate` provides objective/constraint values and gradients;
`Hessian` is needed when Hessian-vector products are requested by second-order methods.

### `Calculate(P, derivatives)`

```python
def Calculate(self, P, derivatives):
    ...
```

Parameters:

- `P`: list of matrices representing the current point, one matrix per manifold block.
- `derivatives`: list of integer flags to request computation of different orders of derivatives.

Possible flags in `derivatives`:

- `0`: compute and store `self.Value`.
- `1`: compute and store `self.Gradient`.
- `2`: forwarded unchanged by the optimization framework.

`Calculate` mutates the object state and returns `void`/`None`.

Flag `2` does **not** request a dense Hessian and does **not** replace `Hessian(K)`.
It is only a second-order request indicator that
the data needed by Hessian-vector products should be prepared.

The derivatives stored here are **Euclidean** derivatives.
Conversion to Riemannian gradients and Hessian is performed by the manifold implementations
and iteration machinery described in [Manifolds.md](Manifolds.md) and [Iterate.md](Iterate.md).

Computation of higher-order derivatives is supposed to reuse the lower-order intermediate variables.

### `Hessian(X)`

```python
def Hessian(self, X):
    ...
```

- Input: list of matrices specifying the vector in which the Hessian is applied for each manifold block.
- Output: list of matrices with the Hessian-vector product block for each manifold block.

Then input and output of each block has the same shape with their corresponding ambient point.

Let

$$
X = (X_1,X_2,\dots,X_n).
$$

The return value represents Euclidean Hessian-vector products

$$
\left(
\sum_{j=1}^{n}
\frac{\partial^2 f}{\partial P_1 \partial P_j}[X_j],
\;
\sum_{j=1}^{n}
\frac{\partial^2 f}{\partial P_2 \partial P_j}[X_j],
\;
\dots
\;,
\sum_{j=1}^{n}
\frac{\partial^2 f}{\partial P_n \partial P_j}[X_j]
\right).
$$

Cross-block second derivatives are therefore included without constructing a dense Hessian matrix.

`Hessian` uses point-dependent data cached by `Calculate` when needed.


## Example

Consider two manifold blocks:

- $x \in \mathrm{Stiefel}(3,1)$
- $y \in \mathbb{R}^{3 \times 1}$

and a fixed matrix $a \in \mathbb{R}^{3 \times 1}$.

Define

$$
f(x,y)=\frac14\sum_i (x_i+y_i-a_i)^4,
$$

together with the equality constraint

$$
g(x,y)=x^\top y=0.
$$

For the objective,

$$
f(x,y)=\frac14\sum_i r_i^4,
$$

define

$$
r=x+y-a.
$$

Then

$$
\nabla f(x,y)=\left(r^{\odot 3},\;r^{\odot 3}\right),
$$

and for a block direction $(u,v)$,
the Euclidean Hessian-vector product is

$$
\nabla^2 f(x,y)[u,v]=\left(3r^{\odot 2}\odot(u+v),\;3r^{\odot 2}\odot(u+v)\right).
$$

Here, $\odot$ and the superscript $\odot k$ denote elementwise multiplication and powers.

For the constraint,

$$
g(x,y)=x^\top y,
$$

the Euclidean gradient is

$$
\nabla g(x,y)=\left(y,\;x\right),
$$

and

$$
\nabla^2 g(x,y)[u,v]=\left(v,\;u\right).
$$

A complete implementation is:

```python
import numpy as np
import Maniverse as mv


class Objective(mv.Function):
	def __init__(self, a):
		super().__init__()
		self.a = a

	def Calculate(self, P, derivatives):
		x, y = P

		if 0 in derivatives:
			# Caches intermediate variable for reuse in Gradient and Hessian.
			self.r = x + y - self.a
			self.Value = 0.25 * np.sum(self.r**4)

		if 1 in derivatives:
			gradient = self.r**3 # Reuses self.r
			self.Gradient = [gradient, gradient]

		# Calculates necessary data for Hessian-vector products instead of the Hessian itself.
		if 2 in derivatives:
			self.hessian_weights = 3 * self.r**2 # Reuses self.r

	def Hessian(self, K):
		u, v = K
		product = self.hessian_weights * (u + v)
		return [product, product]


class OrthogonalityConstraint(mv.Function):
	def __init__(self):
		super().__init__()

	def Calculate(self, P, derivatives):
		x, y = P

		# No need to cache anything for this simple constraint.

		if 0 in derivatives:
			self.Value = float(x.T @ y)

		if 1 in derivatives:
			self.Gradient = [y, x]

	def Hessian(self, K):
		u, v = K
		return [v, u]


a = np.array([[1.0], [0.0], [0.0]])

objective = Objective(a)
constraint = OrthogonalityConstraint()

iterate = mv.Iterate(
	objective,
	[
		mv.Stiefel(np.eye(3, 1)),
		mv.Euclidean(np.zeros((3, 1)))
	],
	[constraint]
)
```

The block order is always the manifold order passed to `mv.Iterate`.
In this example the first block corresponds to the Stiefel variable `x` and the second block corresponds to the Euclidean variable `y`.

The Hessian-vector products include mixed derivatives.
For example, the first objective block receives contributions from both `u` and `v`, and the constraint Hessian maps `(u,v)` to `(v,u)` because the cross-derivatives are nonzero.

The cached data `self.r` helps reduce time cost.
For example, a `Newton` optimizer uses `Calculate(P, [0])`
followed by another `Calculate(P, [1, 2])`.
With the data cached, `self.r` does not need computing in the second run.
By convention, an optimizer will never call `Calculate(P, [1, 2])`
before the necessary cached quantities have been prepared
by an earlier call that includes flag `0`.

Requesting derivative flag `2` prepares the point-dependent weights
`self.hessian_weights` used by `Hessian`.
Optimizers call `Calculate(P, [2])` (or include `2` alongside other flags)
at the relevant point before requesting Hessian-vector products.
Repeated `Hessian` calls reuse these weights.

Membership in the Stiefel manifold is enforced by the manifold block itself, while the additional condition

$$
x^\top y=0
$$

is an explicit equality constraint handled through `OrthogonalityConstraint`.

`mv.Iterate` forms the augmented Lagrangian

$$
f(P)
+
\sum_i \lambda_i g_i(P)
+
\frac{\rho}{2}
\sum_i g_i(P)^2.
$$

Do **not** add multiplier or penalty terms to your objective manually.

See:

- [Iterate.md](Iterate.md)
- [Manifolds.md](Manifolds.md)
- [Optimizers.md](Optimizers.md)

## Pitfalls

- Block order and matrix shapes must match the manifold blocks exactly.
- `Gradient` and `Hessian` provide Euclidean derivatives, not Riemannian derivatives.
- Cached quantities needed by `Hessian` is supposed to be recomputed when a flag `2` is passed to `Calculate`.
- `Hessian` returns Hessian-vector products, not a full dense Hessian matrix.
