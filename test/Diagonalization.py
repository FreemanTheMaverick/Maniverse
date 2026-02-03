import unittest as ut
import numpy as np
import Maniverse as mv

# Symmetric diagonalization
# Finding the eigenvalues and eigenvectors of a symmetric A
# Minimize L(n, C) = || C diag(n) C.t - A ||^2
# A \in Sym(10)
# n \in R(10)
# C \in O(10)

class Obj(mv.Objective):
	def __init__(self):
		super().__init__()
		self.A = np.loadtxt("Sym10.txt", delimiter = ',').reshape([10, 10])

	def Calculate(self, X, derivatives):
		n = self.n = X[0][:, 0]
		C = self.C = X[1]
		if 0 in derivatives:
			self.Value = np.linalg.norm( C * n @ C.T - self.A ) ** 2
		if 1 in derivatives:
			Gn = 2 * ( n - np.diag( C.T @ self.A @ C ) )
			GC = 4 * ( C * n ** 2 - self.A @ C * n )
			self.Gradient = [ Gn, GC ]

	def Hessian(self, V):
		n = self.n
		C = self.C
		delta_n, delta_C = V
		Hnn = 2 * delta_n
		HnC = - 4 * np.diag( C.T @ self.A @ delta_C )
		HCn = 8 * C * n * delta_n[:, 0] - 4 * self.A @ C * delta_n[:, 0]
		HCC = 4 * ( delta_C * n ** 2 - self.A @ delta_C * n )
		return [
					Hnn + HnC,
					HCn + HCC
		]

class TestDiagonalization(ut.TestCase):
	def __init__(self, *args):
		super().__init__(*args)
		self.Obj = Obj()
		self.Manifold0 = mv.Euclidean(np.zeros(10))
		self.Manifold1 = mv.Orthogonal(np.eye(10))
		self.Tolerance = (1.e-5, 1.e-5, 1.e-5)
		self.TrustRegion = mv.TrustRegion()

	def testTruncatedNewton(self):
		M = mv.Iterate(self.Obj, [self.Manifold0, self.Manifold1], True)
		converged = mv.TruncatedNewton(
				M, self.TrustRegion, self.Tolerance,
				0.0001, 28, 0
		)
		assert converged
		assert np.allclose(M.Ms[1].P * M.Ms[0].P[:, 0] @ M.Ms[1].P.T, self.Obj.A)

	def testLBFGS(self):
		M = mv.Iterate(self.Obj, [self.Manifold0, self.Manifold1], True)
		converged = mv.LBFGS(
				M, self.Tolerance,
				100, 110, 0.1, 0.75, 5, 0
		)
		assert converged
		assert np.allclose(M.Ms[1].P * M.Ms[0].P[:, 0] @ M.Ms[1].P.T, self.Obj.A)

if __name__ == "__main__":
	TestDiagonalization().testTruncatedNewton()
	TestDiagonalization().testLBFGS()
