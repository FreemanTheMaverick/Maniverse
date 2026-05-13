import unittest as ut
import numpy as np
import Maniverse as mv

# Rayleigh quotient
# Finding the interior eigenvalue of A
# Minimize L(C) = C.t A C
# A \in Sym(10)
# C \in St(10, 1)

class Obj(mv.Objective):
	def __init__(self):
		super().__init__()
		self.A = np.loadtxt("Sym10.txt", delimiter = ',').reshape([10, 10])

	def Calculate(self, C, derivatives):
		if 0 in derivatives:
			self.Value = np.sum( C[0] * ( self.A @ C[0] ) )
		if 1 in derivatives:
			self.Gradient = [ 2 * self.A @ C[0] ]

	def Hessian(self, V):
		return [ 2 * self.A @ V[0] ]

class TestRayleighInterior(ut.TestCase):
	def __init__(self, *args):
		super().__init__(*args)
		self.Obj = Obj()
		_, Evec = np.linalg.eigh(self.Obj.A)
		self.Manifold = mv.Stiefel( ( Evec[:, 0] + 2 * Evec[:, 1] ) / np.sqrt(5) )
		self.Tolerance = (1.e-5, 1.e-5, 1.e-5)
		self.Solution = Evec[:, 1]

	def testNewtonMR(self):
		M = mv.Iterate(self.Obj, {self.Manifold})
		tr = mv.TrustRegion()
		mr = mv.MinRes(M, 0, 0, (1e-4, 1e-4), M.getDimension(), 0)
		converged = mv.Newton(
				M, tr, mr, self.Tolerance, 5, 0
		)
		assert converged
		assert np.allclose(M.Ms[0].P[:, 0], self.Solution, atol = 1e-5)

	def testLanczos(self):
		M = mv.Iterate(self.Obj, [self.Manifold])
		M.setPoint([self.Solution], 1)
		M.Func.Calculate(M.getPoint(), [0, 1, 2])
		M.setGradient()
		Evals, Evecs = mv.Lanczos(M, M.getDimension(), 0, 0, 0)
		for i in range(len(Evecs)):
			residual = np.linalg.norm( M.ConstraintProjectedHessian(Evecs[i]) - Evals[i] * Evecs[i] )
			assert residual < 1e-5

if __name__ == "__main__":
	TestRayleighInterior().testNewtonCG()
	TestRayleighInterior().testLanczos()
