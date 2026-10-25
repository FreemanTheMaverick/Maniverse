# Manifolds

Built-in manifolds derive from the C++ base class `Manifold` declared in `src/Manifold/Manifold.h`.
Python bindings expose most of the same functionality,
but C++ types and Python representations differ.
Operation availability depends on the concrete manifold implementation;
not every operation declared by the base class
is necessarily functional for every manifold.

## Common `Manifold` interface

### Construction

```cpp
Manifold(Eigen::MatrixXd p, std::string geodesic);
```

```python
Manifold(p, geodesic)
```

The base constructor accepts an ambient `p`
as the current point
and a geodesic-selection string
that instructs `Maniverse` which geodesic operation set
(including retraction and vector transport) is adopted.
Some concrete manifold classes may require extra initialization.

### Data members

| Member | C++ type | Python type | Description |
|--------|----------|-------------|-------------|
| `Name` | `std::string` | `str` | Manifold name string. |
| `Geodesic` | `std::string` | `str` | Geodesic-selection string supported by the concrete manifold. |
| `P` | `Eigen::MatrixXd` | `numpy.ndarray` | Current ambient point on which manifold operations are based. |
| `Ge` | `Eigen::MatrixXd` | `numpy.ndarray` | Stored Euclidean gradient. |
| `Gr` | `Eigen::MatrixXd` | `numpy.ndarray` | Stored Riemannian gradient. |

Although these members are all public,
do not modify them by yourself
unless you are implementing low-level manifolds or optimizers
and understand the consequences.

### Methods

| Method | Description |
|--------|-------------|
| `getDimension()` | Returns the manifold dimension as an `int`. |
| `Inner(X, Y)` | Computes the Riemannian inner product of tangent vectors represented in ambient coordinates. |
| `Retract(X)` | Returns the manifold retraction of tangent vector `X` from the current point `P`. Optimizers use this as the manifold update operation. |
| `InverseRetract(N)` | Returns a tangent vector at `P` pointing toward another point `N` on the same manifold. |
| `TransportTangent(X, Y)` | Transports a tangent vector `X` along another one `Y`. |
| `TransportManifold(X, N)` | Transports a vector from `P` to another manifold point `N` . |
| `TangentProjection(A)` | Projects an ambient vector onto the tangent space at `P`. |
| `TangentPurification(A)` | Applies the tangent-space purification operation defined by the concrete manifold implementation. |
| `setPoint(p, purify)` | Updates `P` to `p`. When `purify=true`, the point is purified or projected according to the manifold's implementation before use. |
| `getGradient()` | Computes the manifold's Riemannian gradient from the stored Euclidean gradient `Ge`, updates `Gr`. |
| `getHessian(HeX, X, weingarten)` | Applies the manifold-specific Riemannian Hessian operation using a Euclidean Hessian-vector product `HeX` and tangent vector `X`. When `weingarten=true`, the Weingarten map is included in the computation; otherwise only the projected Euclidean Hessian term is used.|
| `Clone()` | Returns a copied manifold object. |
| `Share()` | Returns a shared-ownership manifold object suitable for APIs expecting shared pointers. |

## Built-in concrete manifolds

| Manifold class | Point shape / constraints | Dimension |
|----------------|---------------------------|-----------|
| `Euclidean(P0)` | Any matrix shape | Number of entries in `P0` |
| `Stiefel(P0)` | `m × n`, with orthonormal columns (`m ≥ n`) | `mn − n(n+1)/2` |
| `Orthogonal(P0)` | Square orthogonal matrix | `n(n−1)/2` |
| `Flag(P0)` | Orthonormal frame with block sizes configured by `setBlockParameters([...])` | Depends on frame dimensions and block sizes |
| `Grassmann(P0)` | `m × m` orthogonal projector onto the subspace | `k(m−k)`, where `k` is the projector rank |
| `Simplex(P0)` | `n × 1` nonnegative vector whose entries sum to one | `n−1` |

### A note on Grassmann representations

A Grassmann manifold can be represented either
by an orthonormal basis
(with `Flag` and one single number in `setBlockParameters([k])`)
or by the corresponding orthogonal projector (with `Grassmann`).
The projector representation makes the point
independent of the particular basis chosen for the subspace.
The two representation are essentially equivalent.
You may choose either representation based on the needs of your algorithm.

## Product manifolds

Any product manifold of the built-in or user-defined manifolds
can be constructed by combining manifold components into `Iterate` (see [Iterate](Iterate.md)).
Each component retains its own geometry,
while tangent vectors, gradients,
and Hessian operations are applied component-wise.