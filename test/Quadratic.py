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
		self.A = self.A @ self.A + np.eye(10) * 0.01 # Constructing a SPD matrix whose diagonal elements dominate
		for i in range(10):
			for j in range(10):
				if i != j:
					self.A[i, j] *= 0.01
		self.Ax = np.zeros([10, 1]) # Temporary variable to reuse

	def Calculate(self, x, derivatives):
		if 0 in derivatives:
			self.Ax = self.A @ x[0]
			self.Value = np.sum( x[0] * self.Ax )
		if 1 in derivatives:
			self.Gradient = [ 2 * self.Ax ]

	def Hessian(self, v):
		return [ 2 * self.A @ v[0] ]

class PreconObj(UnpreconObj):
	def __init__(self):
		super(PreconObj, self).__init__()
		self.Ainv = np.diag( 1. / ( np.abs( np.diag( 2 * self.A ) ) ) )
		self.Asqrt = np.diag( np.sqrt( np.abs( np.diag( 2 * self.A ) ) ) )
		self.Ainvsqrt = np.linalg.inv( self.Asqrt )

	def Preconditioner(self, V):
		return [ self.Ainv @ V[0] ]

	def PreconditionerSqrt(self, V):
		return [ self.Ainvsqrt @ V[0] ]

	def PreconditionerInvSqrt(self, V):
		return [ self.Asqrt @ V[0] ]

class AndersonObj(UnpreconObj):
	def Calculate(self, x, derivatives):
		super().Calculate(x, derivatives)
		if 1 in derivatives:
			self.Gradient = [ - 2 * self.A @ x[0] ]

class TestQuadratic(ut.TestCase):
	def __init__(self, *args):
		super().__init__(*args)
		self.UnpreconObj = UnpreconObj()
		self.PreconObj = PreconObj()
		self.AndersonObj = AndersonObj()
		self.Manifold = mv.Euclidean(range(10))
		self.Tolerance = (1.e-5, 1.e-5, 1.e-5)

	def testUnpreconNewtonCG(self):
		M = mv.Iterate(self.UnpreconObj, [self.Manifold])
		tr = mv.TrustRegion()
		cg = mv.ConjugateGradient(M, 0, 1, (1e-4, 1e-4), M.getDimension(), 0)
		converged = mv.Newton(
				M, tr, cg, self.Tolerance, 21, 0
		)
		assert converged
		assert np.allclose(M.Ms[0].P, np.zeros_like(M.Ms[0].P), atol = 1e-5)

	def testUnpreconNewtonMR(self):
		M = mv.Iterate(self.UnpreconObj, [self.Manifold])
		tr = mv.TrustRegion()
		mr = mv.MinRes(M, 0, 1, (1e-4, 1e-4), M.getDimension(), 0)
		converged = mv.Newton(
				M, tr, mr, self.Tolerance, 21, 0
		)
		assert converged
		assert np.allclose(M.Ms[0].P, np.zeros_like(M.Ms[0].P), atol = 1e-5)

	def testPreconNewtonCG(self):
		M = mv.Iterate(self.PreconObj, [self.Manifold])
		tr = mv.TrustRegion()
		cg = mv.ConjugateGradient(M, 0, 1, (1e-4, 1e-4), M.getDimension(), 0)
		converged = mv.Newton(
				M, tr, cg, self.Tolerance, 19, 0
		)
		assert converged
		assert np.allclose(M.Ms[0].P, np.zeros_like(M.Ms[0].P), atol = 1e-5)

	def testPreconNewtonMR(self):
		M = mv.Iterate(self.PreconObj, [self.Manifold])
		tr = mv.TrustRegion()
		mr = mv.MinRes(M, 0, 1, (1e-4, 1e-4), M.getDimension(), 0)
		converged = mv.Newton(
				M, tr, mr, self.Tolerance, 20, 0
		)
		assert converged
		assert np.allclose(M.Ms[0].P, np.zeros_like(M.Ms[0].P), atol = 1e-5)

	def testUnpreconLBFGS(self):
		M = mv.Iterate(self.UnpreconObj, [self.Manifold])
		converged = mv.LBFGS(
				M, self.Tolerance,
				20, 11, 0.1, 0.75, 5, 0
		)
		assert converged
		assert np.allclose(M.Ms[0].P, np.zeros_like(M.Ms[0].P), atol = 1e-5)

	def testPreconLBFGS(self):
		M = mv.Iterate(self.PreconObj, [self.Manifold])
		converged = mv.LBFGS(
				M, self.Tolerance,
				20, 7, 0.1, 0.75, 5, 0
		)
		assert converged
		assert np.allclose(M.Ms[0].P, np.zeros_like(M.Ms[0].P), atol = 1e-5)

	def testAnderson(self):
		M = mv.Iterate(self.AndersonObj, [self.Manifold])
		converged = mv.Anderson(
				M, self.Tolerance,
				0.2, 6, 12, 0
		)
		assert converged
		assert np.allclose(M.Ms[0].P, np.zeros_like(M.Ms[0].P), atol = 1e-5)

	def testLanczos(self):
		M = mv.Iterate(self.UnpreconObj, [self.Manifold])
		M.setPoint([np.zeros([10, 1])], 1)
		M.Func.Calculate(M.getPoint(), [0, 1, 2])
		M.setGradient()
		Evals, Evecs = mv.Lanczos(M, M.getDimension(), 0, 0, 0)
		for i in range(len(Evecs)):
			residual = np.linalg.norm( M.ConstraintProjectedHessian(Evecs[i]) - Evals[i] * Evecs[i] )
			assert residual < 1e-5

if __name__ == "__main__":
	TestQuadratic().testUnpreconNewtonCG()
	TestQuadratic().testUnpreconNewtonMR()
	TestQuadratic().testPreconNewtonCG()
	TestQuadratic().testPreconNewtonMR()
	TestQuadratic().testUnpreconLBFGS()
	TestQuadratic().testPreconLBFGS()
	TestQuadratic().testAnderson()
	TestQuadratic().testLanczos()
