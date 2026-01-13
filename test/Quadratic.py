import unittest as ut
import numpy as np
import Maniverse as mv

# Quadratic minimization
# Finding the bottom of a quadratic form
# Minimize L(x) = x.t A x
# A \in SPD(10), nearly diagonal
# x \in R(10)

class UnpreconObj(mv.Objective):
	def __init__(self):
		super().__init__()
		self.A = np.loadtxt("Sym10.txt", delimiter = ',').reshape([10, 10])
		self.A = self.A @ self.A + np.eye(10) * 0.01 # Constructing a SPD matrix
		for i in range(10):
			for j in range(10):
				if i != j:
					self.A[i, j] *= 0.01

	def Calculate(self, X, _):
		self.Value = np.sum( X * ( self.A @ X[0] ) )
		self.Gradient = [2 * self.A @ X[0]]

	def Hessian(self, V):
		return [[ 2 * self.A @ V[0] ]]

class PreconObj(UnpreconObj):
	def __init__(self):
		super(PreconObj, self).__init__()
		self.Ainv = np.diag( 1. / ( np.abs( np.diag( 2 * self.A ) ) ) )
		self.Asqrt = np.diag( np.sqrt( np.abs( np.diag( 2 * self.A ) ) ) )
		self.Ainvsqrt = np.linalg.inv( self.Asqrt )

	def Preconditioner(self, V):
		return [[ self.Ainv @ V[0] ]]

	def PreconditionerSqrt(self, V):
		return [[ self.Ainvsqrt @ V[0] ]]

	def PreconditionerInvSqrt(self, V):
		return [[ self.Asqrt @ V[0] ]]

class TestQuadratic(ut.TestCase):
	def __init__(self, *args):
		super().__init__(*args)
		self.UnpreconObj = UnpreconObj()
		self.PreconObj = PreconObj()
		self.Manifold = mv.Euclidean(range(10))
		self.Tolerance = (1.e-5, 1.e-5, 1.e-5)
		self.TrustRegion = mv.TrustRegion()

	def testUnpreconTruncatedNewton(self):
		M = mv.Iterate(self.UnpreconObj, [self.Manifold.Clone()], True)
		converged = mv.TruncatedNewton(
				M, self.TrustRegion, self.Tolerance,
				0.001, 21, 0
		)
		assert converged
		assert np.allclose(M.Point, np.zeros_like(M.Point), atol = 1e-5)

	def testPreconTruncatedNewton(self):
		M = mv.Iterate(self.PreconObj, [self.Manifold.Clone()], True)
		converged = mv.TruncatedNewton(
				M, self.TrustRegion, self.Tolerance,
				0.001, 19, 0
		)
		assert converged
		assert np.allclose(M.Point, np.zeros_like(M.Point), atol = 1e-5)

	def testUnpreconLBFGS(self):
		M = mv.Iterate(self.UnpreconObj, [self.Manifold.Clone()], True)
		converged = mv.LBFGS(
				M, self.Tolerance,
				20, 11, 0.1, 0.75, 5, 0
		)
		assert converged
		assert np.allclose(M.Point, np.zeros_like(M.Point), atol = 1e-5)

	def testPreconLBFGS(self):
		M = mv.Iterate(self.PreconObj, [self.Manifold.Clone()], True)
		converged = mv.LBFGS(
				M, self.Tolerance,
				20, 7, 0.1, 0.75, 5, 0
		)
		assert converged
		assert np.allclose(M.Point, np.zeros_like(M.Point), atol = 1e-5)

	def testAnderson(self):
		M = mv.Iterate(self.UnpreconObj, [self.Manifold.Clone()], True)
		converged = mv.Anderson(
				M, self.Tolerance,
				0.2, 6, 12, 0
		)
		assert converged
		assert np.allclose(M.Point, np.zeros_like(M.Point), atol = 1e-5)


if __name__ == "__main__":
	TestQuadratic().testUnpreconTruncatedNewton()
	TestQuadratic().testPreconTruncatedNewton()
	TestQuadratic().testUnpreconLBFGS()
	TestQuadratic().testPreconLBFGS()
	TestQuadratic().testAnderson()
