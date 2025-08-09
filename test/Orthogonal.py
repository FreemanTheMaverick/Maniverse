import unittest as ut
import numpy as np
import Maniverse as mv

class Orthogonal(ut.TestCase):

	def testSymmetricDiagonalization(self):
		# Symmetric diagonalization
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
			L = np.linalg.norm( C * n @ C.T - A ) ** 2
			Gn = 2 * ( n - np.diag( C.T @ A @ C ) )
			GC = 4 * ( C * n ** 2 - A @ C * n )
			def Hnn(delta_n):
				return 2 * delta_n
			def HnC(delta_C):
				return - 4 * np.diag( C.T @ A @ delta_C )
			def HCn(delta_n):
				return 8 * C * n * delta_n[:, 0] - 4 * A @ C * delta_n[:, 0]
			def HCC(delta_C):
				return 4 * ( delta_C * n ** 2 - A @ delta_C * n )
			return L, [Gn, GC], [Hnn, HnC, HCn, HCC]
		euclidean = mv.Euclidean(n0)
		orthogonal = mv.Orthogonal(C0)
		M = mv.Iterate([euclidean.Clone(), orthogonal.Clone()], True)
		L = 0
		tr_setting = mv.TrustRegionSetting()
		tol = (1.e-5, 1.e-5, 1.e-5) 
		converged = mv.TrustRegion(
				Objective, tr_setting, tol,
				0.001, 1, 35, L, M, 1
		)
		assert converged
		assert np.allclose(M.Ms[1].P * M.Ms[0].P[:, 0] @ M.Ms[1].P.T, A)

	def testThinSingularValueDecomposition(self):
		# Thin singular value decomposition
		# Finding the singular values and vectors of a rectangular A
		# Minimize L(U, s, V) = || U diag(s) V.t - A ||^2
		# A \in R(10, 6)
		# U \in St(10, 6)
		# s \in R(10)
		# V \in O(6)
		A = np.fromfile("Sym10.dat")[:60]
		A.shape = (10, 6) # Truth value
		U0 = np.eye(10)[:, :6] # Initial guess
		s0 = np.zeros(6)
		V0 = np.eye(6)
		def Objective(Cs, _):
			U = Cs[0]
			s = Cs[1][:, 0]
			V = Cs[2]
			a = U * s @ V.T
			L = np.linalg.norm( a - A )**2
			GU = 2 * ( U * s ** 2 - A @ V * s )
			Gs = 2 * ( s - np.diag( U.T @ A @ V ) )
			GV = 2 * ( V * s ** 2 - A.T @ U * s )
			def HUU(delta_U):
				return 2 * delta_U * s ** 2
			def HUs(delta_s):
				return 4 * U * s * delta_s[:, 0] - 2 * A @ V * delta_s[:, 0]
			def HUV(delta_V):
				return - 2 * A @ delta_V * s
			def HsU(delta_U):
				return - 2 * np.diag( delta_U.T @ A @ V )
			def Hss(delta_s):
				return 2 * delta_s[:, 0]
			def HsV(delta_V):
				return - 2 * np.diag( U.T @ A @ delta_V )
			def HVU(delta_U):
				return - 2 * A.T @ delta_U * s
			def HVs(delta_s):
				return 4 * V * s * delta_s[:, 0] - 2 * A.T @ U * delta_s[:, 0]
			def HVV(delta_V):
				return 2 * delta_V * s ** 2
			return L, [GU, Gs, GV], [
					HUU, HUs, HUV,
					HsU, Hss, HsV,
					HVU, HVs, HVV
			]
		stiefel = mv.Stiefel(U0)
		euclidean = mv.Euclidean(s0)
		orthogonal = mv.Orthogonal(V0)
		M = mv.Iterate([stiefel.Clone(), euclidean.Clone(), orthogonal.Clone()], True)
		L = 0
		tr_setting = mv.TrustRegionSetting()
		tol = (1.e-5, 1.e-5, 1.e-5) 
		converged = mv.TrustRegion(
				Objective, tr_setting, tol,
				0.001, 1, 30, L, M, 0
		)
		assert converged
		assert np.allclose(M.Ms[0].P * M.Ms[1].P[:, 0] @ M.Ms[2].P.T, A)

if __name__ == "__main__":
	Orthogonal().testSymmetricDiagonalization()
	Orthogonal().testThinSingularValueDecomposition()
