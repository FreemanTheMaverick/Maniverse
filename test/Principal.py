import unittest as ut
import numpy as np
import Maniverse as mv

# Principal component analysis
# Finding the space spanned by the highest 5 eigenvectors
# Maximize L(C) = Tr[ C.t A C ]
# A \in Sym(10)
# C \in Flag(1, 2, 3, 4, 5; 10)

class Obj(mv.Objective):
	def __init__(self):
		super().__init__()
		self.A = np.loadtxt("Sym10.txt", delimiter = ',').reshape([10, 10])
	
	def Calculate(self, C, _):
		self.Value = - np.sum( C[0] * ( self.A @ C[0] ) )
		self.Gradient = [ - 2 * self.A @ C[0] ]

	def Hessian(self, V):
		return [ - 2 * self.A @ V[0] ]

class TestPrincipal(ut.TestCase):
	def __init__(self, *args):
		super().__init__(*args)
		self.Obj = Obj()
		self.Manifold = mv.Flag(np.eye(10)[:, 5:]) # Initial guess
		self.Manifold.setBlockParameters([1, 1, 1, 1, 1])
		self.Tolerance = (1.e-5, 1.e-5, 1.e-5)
		self.TrustRegion = mv.TrustRegion()
		self.Solution = np.linalg.eigh(self.Obj.A)[1][:, 5:]

	def testTruncatedNewton(self):
		M = mv.Iterate(self.Obj, [self.Manifold], True)
		converged = mv.TruncatedNewton(
				M, self.TrustRegion, self.Tolerance,
				0.001, 13, 0
		)
		assert converged
		assert np.allclose(M.Ms[0].P @ M.Ms[0].P.T, self.Solution @ self.Solution.T, atol = 1e-5)

	def testLBFGS(self):
		M = mv.Iterate(self.Obj, [self.Manifold], True)
		converged = mv.LBFGS(
				M, self.Tolerance,
				10, 46, 0.1, 0.75, 5, 0
		)
		assert converged
		assert np.allclose(M.Ms[0].P @ M.Ms[0].P.T, self.Solution @ self.Solution.T, atol = 1e-5)

if __name__ == "__main__":
	TestPrincipal().testTruncatedNewton()
	TestPrincipal().testLBFGS()
