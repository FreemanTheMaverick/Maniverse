import unittest as ut
import numpy as np
import Maniverse as mv

# Rayleigh quotient
# Finding the smallest eigenvalue of A
# Minimize L(C) = C.t A C
# A \in Sym(10)
# C \in St(10, 1)

class Obj(mv.Function):
	def __init__(self):
		super().__init__()
		self.A = np.loadtxt("Sym10.txt", delimiter = ',').reshape([10, 10])
		self.C = np.zeros([10, 1])

	def Calculate(self, C_, derivatives):
		C = self.C = C_[0]
		if 0 in derivatives:
			self.Value = np.sum( C * ( self.A @ C ) )
		if 1 in derivatives:
			self.Gradient = [ 2 * self.A @ C ]

	def Hessian(self, V_):
		V = V_[0]
		return [ 2 * self.A @ V ]

class Cons(mv.Function):
	def __init__(self):
		super().__init__()
		self.C = np.zeros([10, 1])

	def Calculate(self, C_, derivatives):
		C = self.C = C_[0]
		if 0 in derivatives:
			self.Value = np.sum( C * C ) - 1
		if 1 in derivatives:
			self.Gradient = [ 2 * C ]

	def Hessian(self, V_):
		V = V_[0]
		return [ 2 * V ]

class TestRayleighLagrange(ut.TestCase):
	def __init__(self, *args):
		super().__init__(*args)
		self.Obj = Obj()
		self.Cons = Cons()
		Eval, Evec = np.linalg.eigh(self.Obj.A)
		self.Manifold = mv.Euclidean( ( Evec[:, 0] + Evec[:, 1] ) / np.sqrt(2) )
		self.Tolerance = (1.e-5, 1.e-5, 1.e-5)
		self.Solution = Evec[:, 0]

	def testNewtonCG(self):
		M = mv.Iterate(self.Obj, {self.Manifold}, {self.Cons})
		tr = mv.TrustRegion()
		cg = mv.ConjugateGradient(M, 0, 1, (1e-4, 1e-4), M.getDimension(), 0)
		converged = mv.AugmentedLagrangian(1, 3.3, 0.8, (1e-5,), 4, 0)(mv.Newton)(
				M, tr, cg, self.Tolerance, 10, 0
		)
		assert converged
		assert np.allclose(M.Manifolds[0].P[:, 0], self.Solution, atol = 1e-5)

	def testNewtonMR(self):
		M = mv.Iterate(self.Obj, {self.Manifold}, {self.Cons})
		tr = mv.TrustRegion()
		mr = mv.MinRes(M, 0, 1, (1e-4, 1e-4), M.getDimension(), 0)
		converged = mv.AugmentedLagrangian(1, 3.3, 0.8, (1e-5,), 4, 0)(mv.Newton)(
				M, tr, mr, self.Tolerance, 10, 0
		)
		assert converged
		assert np.allclose(M.Manifolds[0].P[:, 0], self.Solution, atol = 1e-5)

	def testLBFGS(self):
		M = mv.Iterate(self.Obj, {self.Manifold}, {self.Cons})
		converged = mv.AugmentedLagrangian(1, 3.3, 0.8, (1e-5,), 4, 0)(mv.LBFGS)(
				M, self.Tolerance,
				10, 20, 0.1, 0.75, 7, 0
		)
		assert converged
		assert np.allclose(M.Manifolds[0].P[:, 0], self.Solution, atol = 1e-5)

	def testLanczos(self):
		M = mv.Iterate(self.Obj, [self.Manifold], [self.Cons])
		M.setPoint([self.Solution], 1)
		M.Calculate(M.getPoint(), [0, 1, 2])
		M.setGradient()
		M.setLambda(M.calcLambda())
		Evals, Evecs = mv.Lanczos(M, M.getDimension() - 1, 0, 1, 0)
		for i in range(M.getDimension() - 1):
			residual = np.linalg.norm( M.ConstraintProjection(M.Hessian(Evecs[i]) - Evals[i] * Evecs[i]) )
			assert residual < 1e-5

if __name__ == "__main__":
	TestRayleighLagrange().testNewtonCG()
	TestRayleighLagrange().testNewtonMR()
	TestRayleighLagrange().testLBFGS()
	TestRayleighLagrange().testLanczos()
