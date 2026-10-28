import unittest as ut
import numpy as np
import Maniverse as mv

# Quadratic minimization
# Finding the bottom of a quadratic form
# Minimize L(x) = x.t A x
# A \in SPD(10), nearly diagonal
# x \in R(10)

class Obj(mv.Function):
	def __init__(self):
		super().__init__()
		self.A = np.loadtxt("Sym10.txt", delimiter = ',').reshape([10, 10])
		self.A = self.A @ self.A + np.eye(10) * 0.01 # Constructing a SPD matrix whose diagonal elements dominate
		for i in range(10):
			for j in range(10):
				if i != j:
					self.A[i, j] *= 0.01
		self.Ainv = 1. / ( np.abs( np.diag( 2 * self.A ) ) )
		self.Asqrt = np.sqrt( np.abs( np.diag( 2 * self.A ) ) )
		self.Ainvsqrt = 1. / ( np.sqrt( np.abs( np.diag( 2 * self.A ) ) ) )
		self.Ax = np.zeros([10, 1]) # Temporary variable to reuse

	def Calculate(self, x, derivatives):
		if 0 in derivatives:
			self.Ax = self.A @ x[0]
			self.Value = np.sum( x[0] * self.Ax )
		if 1 in derivatives:
			self.Gradient = [ 2 * self.Ax ]

	def Hessian(self, v):
		return [ 2 * self.A @ v[0] ]

class AndersonObj(Obj):
	def Calculate(self, x, derivatives):
		super().Calculate(x, derivatives)
		if 1 in derivatives:
			self.Gradient = [ - 2 * self.A @ x[0] ]

class TestQuadratic(ut.TestCase):
	def __init__(self, *args):
		super().__init__(*args)
		self.Obj = Obj()
		self.AndersonObj = AndersonObj()
		self.Manifold = mv.Euclidean(range(10))
		self.Tolerance = (1.e-5, 1.e-5, 1.e-5)

	def testUnpreconNewtonCG(self):
		M = mv.Iterate(self.Obj, [self.Manifold])
		tr = mv.TrustRegion()
		cg = mv.ConjugateGradient(1, (1e-4, 1e-4), M.getDimension(), 0)
		mv.initLinearSolverForNewton(cg, M)
		converged = mv.Newton(
				M, tr, cg, self.Tolerance, 21, 0
		)
		assert converged
		assert np.allclose(M.Manifolds[0].P, np.zeros_like(M.Manifolds[0].P), atol = 1e-5)

	def testUnpreconNewtonMR(self):
		M = mv.Iterate(self.Obj, [self.Manifold])
		tr = mv.TrustRegion()
		mr = mv.MinRes(1, (1e-4, 1e-4), M.getDimension(), 0)
		mv.initLinearSolverForNewton(mr, M)
		converged = mv.Newton(
				M, tr, mr, self.Tolerance, 21, 0
		)
		assert converged
		assert np.allclose(M.Manifolds[0].P, np.zeros_like(M.Manifolds[0].P), atol = 1e-5)

	def testPreconNewtonCG(self):
		M = mv.Iterate(self.Obj, [self.Manifold])
		tr = mv.TrustRegion()
		cg = mv.ConjugateGradient(1, (1e-4, 1e-4), M.getDimension(), 0)
		mv.initLinearSolverForNewton(cg, M)
		M.Preconditioner = lambda v : self.Obj.Ainv * v
		converged = mv.Newton(
				M, tr, cg, self.Tolerance, 19, 0
		)
		assert converged
		assert np.allclose(M.Manifolds[0].P, np.zeros_like(M.Manifolds[0].P), atol = 1e-5)

	def testPreconNewtonMR(self):
		M = mv.Iterate(self.Obj, [self.Manifold])
		tr = mv.TrustRegion()
		mr = mv.MinRes(1, (1e-4, 1e-4), M.getDimension(), 0)
		mv.initLinearSolverForNewton(mr, M)
		M.Preconditioner = lambda v : self.Obj.Ainv * v
		converged = mv.Newton(
				M, tr, mr, self.Tolerance, 20, 0
		)
		assert converged
		assert np.allclose(M.Manifolds[0].P, np.zeros_like(M.Manifolds[0].P), atol = 1e-5)

	def testUnpreconLBFGS(self):
		M = mv.Iterate(self.Obj, [self.Manifold])
		converged = mv.LBFGS(
				M, self.Tolerance,
				20, 11, 0.1, 0.75, 5, 0
		)
		assert converged
		assert np.allclose(M.Manifolds[0].P, np.zeros_like(M.Manifolds[0].P), atol = 1e-5)

	def testPreconLBFGS(self):
		M = mv.Iterate(self.Obj, [self.Manifold])
		M.PreconditionerSqrt = lambda v : self.Obj.Ainvsqrt * v
		M.PreconditionerInvSqrt = lambda v : self.Obj.Asqrt * v
		converged = mv.LBFGS(
				M, self.Tolerance,
				20, 7, 0.1, 0.75, 5, 0
		)
		assert converged
		assert np.allclose(M.Manifolds[0].P, np.zeros_like(M.Manifolds[0].P), atol = 1e-5)

	def testAnderson(self):
		M = mv.Iterate(self.AndersonObj, [self.Manifold])
		converged = mv.Anderson(
				M, self.Tolerance,
				0.2, 6, 12, 0
		)
		assert converged
		assert np.allclose(M.Manifolds[0].P, np.zeros_like(M.Manifolds[0].P), atol = 1e-5)

	def testLanczos(self):
		M = mv.Iterate(self.Obj, [self.Manifold])
		M.setPoint([np.zeros([10, 1])], 1)
		M.Calculate(M.getPoint(), [0, 1, 2])
		M.setGradient()
		Evals, Evecs = mv.Lanczos(M, M.getDimension(), 0, 0, 0)
		for i in range(len(Evecs)):
			residual = np.linalg.norm( M.Hessian(Evecs[i]) - Evals[i] * Evecs[i] )
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
