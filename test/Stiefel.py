import unittest as ut
import numpy as np
import Maniverse as mv
from scipy.linalg import expm

class Stiefel(ut.TestCase):

	def testRayleighQuotient(self):
		# Rayleigh quotient
		# Finding the smallest eigenvalue of A
		# Minimize L(C) = C.t A C
		# A \in Sym(10)
		# C \in St(10, 1)
		A = np.fromfile("Sym10.dat")
		A.shape = (10, 10)
		_, evectors = np.linalg.eigh(A)
		Copt = evectors[:, 0] # Truth value
		C0 = ( evectors[:, 0] + evectors[:, 1] ) / np.sqrt(2) # Initial guess
		def Objective(Cs, _):
			C = Cs[0]
			L = np.sum( C * ( A @ C ) )
			G = 2 * A @ C
			def H(v):
				return 2 * A @ v
			return L, [G], [H]
		stiefel = mv.Stiefel(C0)
		M = mv.Iterate({stiefel.Clone()}, True)
		L = 0
		tr_setting = mv.TrustRegionSetting()
		tol = (1.e-5, 1.e-5, 1.e-5) 
		converged = mv.TrustRegion(
				Objective, tr_setting, tol,
				0.001, 1, 3, L, M, 0
		)
		assert converged
		assert np.allclose(M.Point.T, Copt)

	def testRayleighQuotientLBFGS(self):
		# Rayleigh quotient
		# Finding the smallest eigenvalue of A
		# Minimize L(C) = C.t A C
		# A \in Sym(10)
		# C \in St(10, 1)
		A = np.fromfile("Sym10.dat")
		A.shape = (10, 10)
		_, evectors = np.linalg.eigh(A)
		Copt = evectors[:, 0] # Truth value
		C0 = ( evectors[:, 0] + evectors[:, 1] ) / np.sqrt(2) # Initial guess
		def Objective(Cs, _):
			C = Cs[0]
			L = np.sum( C * ( A @ C ) )
			G = 2 * A @ C
			return L, [G]
		stiefel = mv.Stiefel(C0)
		M = mv.Iterate({stiefel.Clone()}, True)
		L = 0
		tol = (1.e-5, 1.e-5, 1.e-5)
		converged = mv.LBFGS(
				Objective, tol,
				20, 12, L, M, 0
		)
		assert converged
		assert np.allclose(M.Point.T, Copt)

	def testOrthogonalProjection(self):
		# Orthogonal projection
		# Finding the Stiefel matrix closest to the given matrix A
		# Minimize L(C) = || C - A ||^2
		# A \in R(10, 6)
		# C \in St(10, 6)
		A = np.fromfile("Sym10.dat")[:60]
		A.shape = (10, 6)
		U, _, Vt = np.linalg.svd(A, full_matrices = False)
		Copt = U @ Vt # Truth value
		C0 = U.copy() # Initial guess
		def Objective(Cs, _):
			C = Cs[0]
			L = np.linalg.norm(C - A) ** 2
			G = 2 * ( C - A )
			def H(v):
				return 2 * v
			return L, [G], [H]
		stiefel = mv.Stiefel(C0)
		M = mv.Iterate({stiefel.Clone()}, True)
		L = 0
		tr_setting = mv.TrustRegionSetting()
		tol = (1.e-5, 1.e-5, 1.e-5)
		converged = mv.TrustRegion(
				Objective, tr_setting, tol,
				0.001, 1, 15, L, M, 0
		)
		assert converged
		assert np.allclose(M.Point, Copt)

	def testOrthogonalProjectionLBFGS(self):
		# Orthogonal projection
		# Finding the Stiefel matrix closest to the given matrix A
		# Minimize L(C) = || C - A ||^2
		# A \in R(10, 6)
		# C \in St(10, 6)
		A = np.fromfile("Sym10.dat")[:60]
		A.shape = (10, 6)
		U, _, Vt = np.linalg.svd(A, full_matrices = False)
		Copt = U @ Vt # Truth value
		C0 = U.copy() # Initial guess
		def Objective(Cs, _):
			C = Cs[0]
			L = np.linalg.norm(C - A) ** 2
			G = 2 * ( C - A )
			return L, [G]
		stiefel = mv.Stiefel(C0)
		M = mv.Iterate({stiefel.Clone()}, True)
		L = 0
		tol = (1.e-5, 1.e-5, 1.e-5)
		converged = mv.LBFGS(
				Objective, tol,
				20, 35, L, M, 0
		)
		assert converged
		assert np.allclose(M.Point, Copt, atol = 1.e-5)

	def testOrthogonalProjectionAnderson(self):
		# Orthogonal projection
		# Finding the Stiefel matrix closest to the given matrix A
		# Minimize L(C) = || C - A ||^2
		# A \in R(10, 6)
		# C \in St(10, 6)
		A = np.fromfile("Sym10.dat")[:60]
		A.shape = (10, 6)
		U, _, Vt = np.linalg.svd(A, full_matrices = False)
		Copt = U @ Vt # Truth value
		C0 = Copt @ expm(A[:6, :] - A[:6, :].T) # Initial guess
		def Objective(Cs):
			C = Cs[0]
			L = np.linalg.norm(C - A) ** 2
			G = 2 * ( C - A )
			return L, [- 0.2 * G]
		stiefel = mv.Stiefel(C0)
		M = mv.Iterate({stiefel.Clone()}, True)
		L = 0
		tol = (1.e-5, 1.e-5, 1.e-5)
		converged = mv.Anderson(
				Objective, tol, 1,
				6, 18, L, M, 0
		)
		assert converged
		assert np.allclose(M.Point, Copt, atol = 1.e-5)

if __name__ == "__main__":
	Stiefel().testRayleighQuotient()
	Stiefel().testRayleighQuotientLBFGS()
	Stiefel().testOrthogonalProjection()
	Stiefel().testOrthogonalProjectionLBFGS()
	Stiefel().testOrthogonalProjectionAnderson()
