import unittest as ut
import numpy as np
import Maniverse as mv

class Orthogonal(ut.TestCase):

	def testSymmetricDiagonalization(self):
		# Symemtric diagonalization
		# Finding the eigenvalues and eigenvectors of a symmetric A
		# Minimize L(n, C) = || C diag(n) C.t - A ||^2
		# A \in Sym(10)
		# n \in R(10)
		# C \in Orthogonal(10)
		A = np.fromfile("Sym10.dat")
		A.shape = (10, 10) # Truth value
		n0 = np.zeros(10) # Initial guess
		C0 = np.eye(10)
		def Objective(Cs, _):
			n = Cs[0][:, 0]
			C = Cs[1]
			a = C * n @ C.T
			L = np.linalg.norm( a - A )**2
			Ga = 2 * ( a - A )
			Gn = np.diag( C.T @ Ga @ C )
			GC = 2 * Ga @ C * n
			def Hnn(delta_n):
				return 2 * delta_n
			def HnC(delta_C):
				part1 = 4 * np.diag( C.T @ delta_C * n )
				part2 = 2 * np.sum( delta_C.T @ Ga @ C, axis = 1 )
				return part1 + part2
			def HCn(delta_n):
				part1 = 4 *  C * delta_n[:, 0] * n
				part2 = 2 * Ga @ C * delta_n[:, 0]
				return part1 + part2
			def HCC(delta_C):
				part1 = 2 * delta_C * n
				part2 = 2 * C * n @ delta_C.T @ C
				part3 = Ga @ delta_C
				return 2 * ( part1 + part2 + part3 ) * n
			return L, [Gn, GC], [Hnn, HnC, HCn, HCC]
		euclidean = mv.Euclidean(n0)
		orthogonal = mv.Orthogonal(C0)
		M = mv.Iterate([euclidean.Clone(), orthogonal.Clone()], True)
		L = 0
		tr_setting = mv.TrustRegionSetting()
		tol = (1.e-5, 1.e-5, 1.e-5) 
		converged = mv.TrustRegion(
				Objective, tr_setting, tol,
				0.001, 1, 35, L, M, 0
		)
		assert converged
		assert np.allclose(M.Ms[1].P * M.Ms[0].P[:, 0] @ M.Ms[1].P.T, A)

if __name__ == "__main__":
	Orthogonal().testSymmetricDiagonalization()
