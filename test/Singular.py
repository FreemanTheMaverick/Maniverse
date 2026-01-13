import unittest as ut
import numpy as np
import Maniverse as mv

# Thin singular value decomposition
# Finding the singular values and vectors of a rectangular A
# Minimize L(U, s, V) = || U diag(s) V.t - A ||^2
# A \in R(10, 6)
# U \in St(10, 6)
# s \in R(10)
# V \in O(6)

class Obj(mv.Objective):
	def __init__(self):
		super().__init__()
		self.A = np.loadtxt("Sym10.txt", delimiter = ',')[:60].reshape([10, 6])

	def Calculate(self, X, _):
		U = self.U = X[0]
		s = self.s = X[1][:, 0]
		V = self.V = X[2]
		a = U * s @ V.T
		self.Value = np.linalg.norm( a - self.A )**2
		GU = 2 * ( U * s ** 2 - self.A @ V * s )
		Gs = 2 * ( s - np.diag( U.T @ self.A @ V ) )
		GV = 2 * ( V * s ** 2 - self.A.T @ U * s )
		self.Gradient = [ GU, Gs, GV ]

	def Hessian(self, K):
		U = self.U
		s = self.s
		V = self.V
		delta_U, delta_s, delta_V = K
		HUU = 2 * delta_U * s ** 2
		HUs = 4 * U * s * delta_s[:, 0] - 2 * self.A @ V * delta_s[:, 0]
		HUV = - 2 * self.A @ delta_V * s
		HsU = - 2 * np.diag( delta_U.T @ self.A @ V )
		Hss = 2 * delta_s[:, 0]
		HsV = - 2 * np.diag( U.T @ self.A @ delta_V )
		HVU = - 2 * self.A.T @ delta_U * s
		HVs = 4 * V * s * delta_s[:, 0] - 2 * self.A.T @ U * delta_s[:, 0]
		HVV = 2 * delta_V * s ** 2
		return [
					[ HUU, HUs, HUV ],
					[ HsU, Hss, HsV ],
					[ HVU, HVs, HVV ]
		]

class TestSingular(ut.TestCase):
	def __init__(self, *args):
		super().__init__(*args)
		self.Obj = Obj()
		self.Manifold0 = mv.Stiefel(np.eye(10)[:, :6])
		self.Manifold1 = mv.Euclidean(np.zeros(6))
		self.Manifold2 = mv.Orthogonal(np.eye(6))
		self.Tolerance = (1.e-5, 1.e-5, 1.e-5)
		self.TrustRegion = mv.TrustRegion()

	def testTruncatedNewton(self):
		M = mv.Iterate(self.Obj, [self.Manifold0.Clone(), self.Manifold1.Clone(), self.Manifold2.Clone()], True)
		converged = mv.TruncatedNewton(
				M, self.TrustRegion, self.Tolerance,
				0.001, 25, 0
		)
		assert converged
		assert np.allclose(M.Ms[0].P * M.Ms[1].P[:, 0] @ M.Ms[2].P.T, self.Obj.A)

	def testLBFGS(self):
		M = mv.Iterate(self.Obj, [self.Manifold0.Clone(), self.Manifold1.Clone(), self.Manifold2.Clone()], True)
		converged = mv.LBFGS(
				M, self.Tolerance,
				100, 125, 0.1, 0.75, 5, 0
		)
		assert converged
		assert np.allclose(M.Ms[0].P * M.Ms[1].P[:, 0] @ M.Ms[2].P.T, self.Obj.A)

if __name__ == "__main__":
	TestSingular().testTruncatedNewton()
	TestSingular().testLBFGS()
