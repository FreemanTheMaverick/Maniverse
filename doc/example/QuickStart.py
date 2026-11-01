import unittest as ut

import numpy as np
import Maniverse as mv


A = np.array([
    [4.0, 1.0, 0.5, 0.0],
    [1.0, 3.0, 0.0, 0.5],
    [0.5, 0.0, 2.0, 0.5],
    [0.0, 0.5, 0.5, 1.0],
])


class PrincipalSubspace(mv.Function):
	def Calculate(self, points, derivatives):
		C = points[0]
		if 0 in derivatives:
			self.Value = -np.sum(C * (A @ C))
		if 1 in derivatives:
			self.Gradient = [-2.0 * A @ C]

	def Hessian(self, directions):
		return [-2.0 * A @ directions[0]]


class TestQuickStart(ut.TestCase):
	def test_principal_subspace(self):
		C0 = np.eye(4)[:, :2]
		manifold = mv.Flag(C0)
		manifold.setBlockParameters([2])
		objective = PrincipalSubspace()
		M = mv.Iterate(objective, [manifold])

		tr = mv.TrustRegion()
		cg = mv.ConjugateGradient(
			True, [1e-4, 1e-4], M.getDimension(), False
		)
		mv.initLinearSolverForNewton(cg, M)
		converged = mv.Newton(
			M, tr, cg, [1e-6, 1e-6, 1e-6], 20, 0
		)

		C = M.getPoint()[0]
		_, eigenvectors = np.linalg.eigh(A)
		C_exact = eigenvectors[:, -2:]

		self.assertTrue(converged)
		np.testing.assert_allclose(C.T @ C, np.eye(2), atol=1e-6)
		np.testing.assert_allclose(
			C @ C.T, C_exact @ C_exact.T, atol=1e-5
		)


if __name__ == "__main__":
	ut.main()