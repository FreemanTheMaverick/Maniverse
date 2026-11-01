# QuickStart

This example finds the two-dimensional principal subspace of a symmetric matrix.
We maximize

$$
\mathrm{Tr}(C^T A C), \qquad C^T C=I,
$$

by minimizing its negative on the Grassmann manifold $\mathrm{Gr}(2,4)$,
represented by `Flag` in `Maniverse`.
Unlike an unconstrained Euclidean example,
this demonstrates how `Maniverse` keeps every iterate on a nonlinear manifold.
We use Newton's method with a trust region as the example optimizer
and conjugate gradient to solve its inner subproblems.

This is a small, self-contained version of `test/Principal.cpp` and `test/Principal.py`.

## Step 0 — Pre-requisites

* You have installed the Python extension or built the C++ library (see [Installation](Installation.md)).

## Step 1 — Define the principal-subspace objective

The objective function is defined as a derived class of the built-in `Function` class (see [Function](Function.md)).

**Python — `QuickStart.py`**

```python
import numpy as np
import Maniverse as mv

A = np.array([
    [4.0, 1.0, 0.5, 0.0],
    [1.0, 3.0, 0.0, 0.5],
    [0.5, 0.0, 2.0, 0.5],
    [0.0, 0.5, 0.5, 1.0],
])

class PrincipalSubspace(mv.Function):
	def Calculate(self, points, derivatives):
		C = points[0]
		if 0 in derivatives:
			self.Value = -np.sum(C * (A @ C))
		if 1 in derivatives:
			self.Gradient = [-2.0 * A @ C]

	def Hessian(self, directions):
		return [-2.0 * A @ directions[0]]
```

**C++ — `quickstart.cpp`**

```cpp
#include <Maniverse/Manifold/Flag.h>
#include <Maniverse/LinearSolver/ConjugateGradient.h>
#include <Maniverse/Optimizer/Newton.h>
#include <Eigen/Dense>
#include <algorithm>
#include <cstdio>
#include <vector>

namespace mv = Maniverse;

class PrincipalSubspace : public mv::Function {
public:
    Eigen::MatrixXd A;

	PrincipalSubspace() : A(4, 4) {
		A << 4.0, 1.0, 0.5, 0.0,
		     1.0, 3.0, 0.0, 0.5,
		     0.5, 0.0, 2.0, 0.5,
		     0.0, 0.5, 0.5, 1.0;
	}

	void Calculate(std::vector<Eigen::MatrixXd> points,
	               std::vector<int> derivatives) override {
		const Eigen::MatrixXd &C = points[0];
		if (std::count(derivatives.begin(), derivatives.end(), 0))
			Value = -C.cwiseProduct(A * C).sum();
		if (std::count(derivatives.begin(), derivatives.end(), 1))
			Gradient = {-2.0 * A * C};
	}

	std::vector<Eigen::MatrixXd> Hessian(
			std::vector<Eigen::MatrixXd> directions) const override {
		return {-2.0 * A * directions[0]};
	}
};
```

The minus sign turns maximization into minimization.
`Calculate` supplies the ambient objective and gradient;
`Hessian` supplies a Hessian-vector product without constructing a dense Hessian.
Both gradient and Hessian provided by users are of the Euclidean space.
The manifold projects derivatives into the appropriate tangent space.

## Step 2 — Build the Grassmann manifold and `Iterate`

**Python — continue `QuickStart.py`**

```python
C0 = np.eye(4)[:, :2]
manifold = mv.Flag(C0)
manifold.setBlockParameters([2])
objective = PrincipalSubspace()
M = mv.Iterate(objective, [manifold])
```

**C++ — append inside `main()`**

```cpp
int main() {
	Eigen::MatrixXd C0 = Eigen::MatrixXd::Identity(4, 2);
	mv::Flag manifold(C0);
	manifold.setBlockParameters({2});
	PrincipalSubspace objective;
	mv::Iterate M(objective, {manifold.Share()});
```

`C0` is the initial guess for the optimization.
Its columns form an orthonormal basis for the initial two-dimensional subspace.
`Flag` with one block of size 2 represents $\mathrm{Gr}(2,4)$.
Rotating the columns of $C$ does not change the represented subspace,
so the optimizer seeks the subspace itself rather than a particular basis.

`Iterate` combines the objective and manifold state to be consumed by
the optimizers, linear solvers, and diagonalization routine
(see [Iterate](Iterate.md)).

## Step 3 — Run Newton with a trust region

**Python — continue `QuickStart.py`**

```python
tr = mv.TrustRegion()
cg = mv.ConjugateGradient(True, [1e-4, 1e-4], M.getDimension(), False)
mv.initLinearSolverForNewton(cg, M)
converged = mv.Newton(M, tr, cg, [1e-6, 1e-6, 1e-6], 20, 1)
C = M.getPoint()[0]
print("converged:", converged)
print("C^T C =\n", C.T @ C)
```

**C++ — continue inside `main()`**

```cpp
    mv::TrustRegion tr;
	mv::ConjugateGradient cg(1, {1e-4, 1e-4}, M.getDimension(), 0);
	mv::initLinearSolverForNewton(cg, M);
	const bool converged = mv::Newton(
		M, tr, cg, {1e-6, 1e-6, 1e-6}, 20, 1
	);
	const Eigen::MatrixXd C = M.getPoint()[0];

	std::printf("converged = %d\n", converged);
	std::printf("C^T C =\n");
	std::cout << C.transpose() * C << '\n';
```

`tr` configures the trust region,
which limits each Newton step to an area
where the local quadratic model is considered reliable.
`cg` is the conjugate-gradient linear solver that
approximately solves the Newton equation using Hessian-vector products;
`M.getDimension()` sets the size of that linear problem.
`initLinearSolverForNewton(cg, M)` then connects `cg` to this particular `Iterate`
by providing its manifold inner product, tangent projection,
Hessian operator, and preconditioner.
It must be called before `Newton`.

`Maniverse` computes the resulting tangent steps
and retracts them back to the manifold.
Consequently, the final columns remain orthonormal
even though the optimizer works with unconstrained ambient derivatives.

## Step 4 — Verify the resulting subspace

Eigenvectors are not unique (signs may flip),
and any orthogonal rotation of the two columns gives an equivalent basis.
Compare the projection matrices $CC^T$ instead.

**Python — continue `QuickStart.py`**

```python
_, eigenvectors = np.linalg.eigh(A)
C_exact = eigenvectors[:, -2:]
projector_error = np.linalg.norm(C @ C.T - C_exact @ C_exact.T)
print("projector error:", projector_error)
```

**C++ — continue inside `main()`**

```cpp
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(objective.A);
	const Eigen::MatrixXd C_exact = eig.eigenvectors().rightCols(2);
	const double projector_error =
		(C * C.transpose() - C_exact * C_exact.transpose()).norm();
	std::printf("projector error = %.3e\n", projector_error);
    return converged && projector_error < 1e-5 ? 0 : 1;
}
```

A small projector error confirms that Maniverse found the subspace spanned by the two largest-eigenvalue eigenvectors of $A$.

## Step 5 — (Optional) Inspect the Riemannian Hessian

At a local minimum, the Riemannian Hessian should have no negative eigenvalues.
Lanczos can inspect its spectrum.

Now Hessian diagonalization is available and functional,
but it is still under development.
Its APIs may change in the near future.
Check `test/` for usage of diagonalization.

**Python — finish `QuickStart.py`**

**C++ — finish `main()`**

```cpp
    M.Calculate(M.getPoint(), {0, 1, 2});
	M.setGradient();
	const auto [Evals, Evecs] =
		mv::Lanczos(M, M.getDimension(), 0, 0, 1);
	std::printf("smallest Hessian eigenvalue = %.6e\n", Evals[0]);

```

A substantially negative value indicates a direction of negative curvature; a nonnegative value is consistent with a local minimum.

## Step 6 — Build and run the C++ example

Save the Python fragments above in order as `QuickStart.py`,
then run:

```bash
python3 QuickStart.py
```

Save the C++ fragments above in order as `quickstart.cpp`,
compile and run it:

```bash
g++ quickstart.cpp -std=c++17 -I"$MANIVERSE/include" \
    -L"$MANIVERSE/lib" -l:libmaniverse.a -o quickstart
./quickstart
```

`$MANIVERSE` is used here to locate the generated headers and library.
If Eigen is not installed in a compiler-default include directory,
also pass its include directory, for example `-isystem /path/to/eigen`.