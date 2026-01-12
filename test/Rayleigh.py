import unittest as ut
import numpy as np
import Maniverse as mv

# Rayleigh quotient
# Finding the smallest eigenvalue of A
# Minimize L(C) = C.t A C
# A \in Sym(10)
# C \in St(10, 1)

class Obj(mv.Objective):
	def __init__(self):
		super().__init__()
		self.A = np.fromfile("Sym10.dat").reshape([10, 10])

	def Calculate(self, C, _):
		self.Value = np.sum( C[0] * ( self.A @ C[0] ) )
		self.Gradient = [ 2 * self.A @ C[0] ]

	def Hessian(self, X):
		return [[ 2 * self.A @ X[0] ]]

class TestRayleigh(ut.TestCase):
	def __init__(self, *args):
		super().__init__(*args)
		self.Obj = Obj()
		_, Evec = np.linalg.eigh(self.Obj.A)
		self.Manifold = mv.Stiefel( ( Evec[:, 0] + Evec[:, 1] ) / np.sqrt(2) )
		self.Tolerance = (1.e-5, 1.e-5, 1.e-5)
		self.TrustRegion = mv.TrustRegion()
		self.Solution = Evec[:, 0]

	def testTruncatedNewton(self):
		M = mv.Iterate(self.Obj, {self.Manifold.Clone()}, True)
		tol = (1.e-5, 1.e-5, 1.e-5) 
		converged = mv.TruncatedNewton(
				M, self.TrustRegion, self.Tolerance,
				0.001, 3, 0
		)
		assert converged
		assert np.allclose(M.Point[:, 0], self.Solution, atol = 1e-5)

	def testLBFGS(self):
		M = mv.Iterate(self.Obj, {self.Manifold.Clone()}, True)
		converged = mv.LBFGS(
				M, self.Tolerance,
				10, 8, 0.1, 0.75, 5, 0
		)
		assert converged
		assert np.allclose(M.Point[:, 0], self.Solution, atol = 1e-5)

if __name__ == "__main__":
	TestRayleigh().testTruncatedNewton()
	TestRayleigh().testLBFGS()
