import unittest as ut
import numpy as np
import Maniverse as mv
from scipy.linalg import expm

# Orthogonal projection
# Finding the Stiefel matrix closest to the given matrix A
# Minimize L(C) = || C - A ||^2
# A \in R(10, 6)
# C \in St(10, 6)

class Obj(mv.Objective):
	def __init__(self):
		super().__init__()
		self.A = np.loadtxt("Sym10.txt", delimiter = ',')[:60].reshape([6, 10]).T

	def Calculate(self, C, _):
		self.Value = np.linalg.norm(C[0] - self.A) ** 2
		self.Gradient = [ 2 * ( C[0] - self.A ) ]

	def Hessian(self, X):
		return [[ 2 * X[0] ]]

class AndersonObj(Obj):
	def Calculate(self, C, _):
		super().Calculate(C, _)
		self.Gradient = [ -2 * ( C[0] - self.A ) ]

class TestProjection(ut.TestCase):
	def __init__(self, *args):
		super().__init__(*args)
		self.Obj = Obj()
		self.AndersonObj = AndersonObj()
		U, _, Vt = np.linalg.svd(self.Obj.A, full_matrices = False)
		self.Manifold = mv.Stiefel( U @ Vt @ expm( self.Obj.A[4:, :] - self.Obj.A[4:, :].T ) )
		self.Solution = U @ Vt
		self.Tolerance = (1.e-5, 1.e-5, 1.e-5)
		self.TrustRegion = mv.TrustRegion()

	def testTruncatedNewton(self):
		M = mv.Iterate(self.Obj, [self.Manifold], True)
		converged = mv.TruncatedNewton(
				M, self.TrustRegion, self.Tolerance,
				0.001, 9, 0
		)
		assert converged
		assert np.allclose(M.Point, self.Solution, atol = 1e-5)

	def testLBFGS(self):
		M = mv.Iterate(self.Obj, [self.Manifold], True)
		converged = mv.LBFGS(
				M, self.Tolerance,
				20, 19, 0.1, 0.75, 5, 0
		)
		assert converged
		assert np.allclose(M.Point, self.Solution, atol = 1.e-5)

	def testAnderson(self):
		M = mv.Iterate(self.AndersonObj, [self.Manifold], True)
		converged = mv.Anderson(
				M, self.Tolerance,
				0.2, 6, 28, 0
		)
		assert converged
		assert np.allclose(M.Point, self.Solution, atol = 1.e-5)

if __name__ == "__main__":
	TestProjection().testTruncatedNewton()
	TestProjection().testLBFGS()
	TestProjection().testAnderson()
