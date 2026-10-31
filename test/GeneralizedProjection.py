import unittest as ut
import numpy as np
import Maniverse as mv
from scipy.linalg import sqrtm, expm

# Orthogonal projection
# Finding the generalized Stiefel matrix closest to the given matrix A
# Minimize L(C) = || C - A ||^2_B
# A \in R(10, 6)
# B \in SPD(10)
# C \in St(10, 6)

class Obj(mv.Function):
	def __init__(self):
		super().__init__()
		self.A = np.loadtxt("Sym10.txt", delimiter = ',')[:60].reshape([6, 10]).T
		self.B = np.loadtxt("Sym10.txt", delimiter = ',').reshape([10, 10]).T
		self.B = sqrtm(self.B @ self.B)

	def Calculate(self, C, derivatives):
		C_minus_A = C[0] - self.A
		if 0 in derivatives:
			self.Value = ( C_minus_A * ( self.B @ C_minus_A ) ).sum()
		if 1 in derivatives:
			self.Gradient = [ 2 * self.B @ C_minus_A ]

	def Hessian(self, X):
		return [ 2 * self.B @ X[0] ]

class TestProjection(ut.TestCase):
	def __init__(self, *args):
		super().__init__(*args)
		self.Obj = Obj()
		L = np.linalg.cholesky(self.Obj.B)
		U, _, Vt = np.linalg.svd(L.T @ self.Obj.A, full_matrices = False)
		self.Solution = np.linalg.inv(L.T) @ U @ Vt
		self.Manifold = mv.GeneralizedStiefel( self.Solution @ expm( self.Obj.A[4:, :] - self.Obj.A[4:, :].T ), self.Obj.B )
		self.Tolerance = (1.e-5, 1.e-5, 1.e-5)

	def testNewtonCG(self):
		M = mv.Iterate(self.Obj, [self.Manifold])
		tr = mv.TrustRegion()
		cg = mv.ConjugateGradient(1, (1e-4, 1e-4), M.getDimension(), 0)
		mv.initLinearSolverForNewton(cg, M)
		converged = mv.Newton(
				M, tr, cg, self.Tolerance, 9, 0
		)
		assert converged
		assert np.allclose(M.Manifolds[0].P, self.Solution, atol = 1e-5)

	def testNewtonMR(self):
		M = mv.Iterate(self.Obj, [self.Manifold])
		tr = mv.TrustRegion()
		mr = mv.MinRes(1, (1e-4, 1e-4), M.getDimension(), 0)
		mv.initLinearSolverForNewton(mr, M)
		converged = mv.Newton(
				M, tr, mr, self.Tolerance, 10, 0
		)
		assert converged
		assert np.allclose(M.Manifolds[0].P, self.Solution, atol = 1e-5)

	def testLBFGS(self):
		M = mv.Iterate(self.Obj, [self.Manifold])
		converged = mv.LBFGS(
				M, self.Tolerance,
				20, 35, 0.1, 0.75, 5, 0
		)
		assert converged
		assert np.allclose(M.Manifolds[0].P, self.Solution, atol = 1e-5)

	def testLanczos(self):
		M = mv.Iterate(self.Obj, [self.Manifold])
		M.setPoint([self.Solution], 1)
		M.Calculate(M.getPoint(), [0, 1, 2])
		M.setGradient()
		Evals, Evecs = mv.Lanczos(M, M.getDimension(), 0, 0, 0)
		for i in range(len(Evecs)):
			residual = np.linalg.norm( M.Hessian(Evecs[i]) - Evals[i] * Evecs[i] )
			assert residual < 1e-5

if __name__ == "__main__":
	TestProjection().testNewtonCG()
	TestProjection().testNewtonMR()
	TestProjection().testLBFGS()
	TestProjection().testLanczos()
