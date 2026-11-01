# Maniverse
> Numerical optimization on manifolds

## Overview
`Maniverse` is a C++17 library for numerical optimization on manifolds,
with a Python interface.
It was developed for the quantum chemistry packages
[Chinium](https://github.com/FreemanTheMaverick/Chinium) and
[Orbaplaw](https://github.com/FreemanTheMaverick/Orbaplaw),
and is also designed for use in other applications.
It includes built-in manifolds and optimization algorithms,
and supports extensions for additional problem types.

## Features
- **Manifolds**: (Generalized) Stiefel and related manifolds (Flag, Grassmann, and Orthogonal), the Simplex manifold, and user-defined manifolds. Products of manifolds are supported.
- **Optimizers**: Trust-region Newton's method with conjugate-gradient or minimum-residual inner solvers, L-BFGS, Anderson acceleration, and an augmented-Lagrangian wrapper for equality constraints.
- **Linear solvers**: Conjugate Gradient and MinRes, with preconditioning and optional handling of non-positive curvature in Newton's trust-region subproblem.
- **Hessian eigenpairs**: A Lanczos method for examining the stability of constrained and unconstrained solutions.

## Guides
Use these guides to install `Maniverse`, formulate your optimization problem, choose an appropriate solver, and solve it with the library.
- [Installation](doc/Installation.md)
- [Quick start: Rayleigh quotient on the Stiefel manifold](doc/QuickStart.md)
- [Manifolds: base class and built-ins](doc/Manifolds.md)
- [Objective and constraint functions](doc/Function.md)
- [The Iterate class that ties manifolds and functions together](doc/Iterate.md)
- [Optimizers: Newton/trust-region, L-BFGS, Anderson, Augmented Lagrangian](doc/Optimizers.md)
- [Linear system solvers: CG and MinRes](doc/LinearSolvers.md)
- [Diagonalizer: Lanczos method](doc/Diagonalizers.md)
- [FAQ: purpose, motivation, and maintenance](doc/FAQ.md)

## Related links

- [Maniverse on GitHub](https://github.com/FreemanTheMaverick/Maniverse)
- [Chinium](https://github.com/FreemanTheMaverick/Chinium), a quantum chemistry package that uses `Maniverse`
- [Manopt](https://www.manopt.org/), a related manifold optimization toolbox