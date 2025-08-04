import unittest as ut
import numpy as np
import Maniverse as mv

# L(C) = Tr[ C.t A C ]
# A \in Sym(10)
# C \in Flag(10, 6)

class Flag(ut.TestCase):

	def testPrincipalComponentAnalysis(self):
		# Principal component analysis
		# Finding the space spanned by the highest 5 eigenvectors
		# Maximize L(C) = Tr[ C.t A C ]
		# A \in Sym(10)
		# C \in Flag(1, 2, 3, 4, 5; 10)
		A = np.fromfile("Sym10.dat")
		A.shape = (10, 10)
		_, evectors = np.linalg.eigh(A)
		Copt = evectors[:, 5:] # Truth value
		C0 = np.eye(10)[:, 5:] # Initial guess
		def Objective(Cs, _):
			C = Cs[0]
			L = - np.sum( C * ( A @ C ) )
			G = - 2 * A @ C
			def H(v):
				return - 2 * A @ v
			return L, [G], [H]
		flag = mv.Flag(C0)
		flag.setBlockParameters([1, 1, 1, 1, 1])
		M = mv.Iterate({flag.Clone()}, True)
		L = 0
		tr_setting = mv.TrustRegionSetting()
		tol = (1.e-5, 1.e-5, 1.e-5) 
		converged = mv.TrustRegion(
				Objective, tr_setting, tol,
				0.001, 1, 100, L, M, 0
		)
		assert converged
		assert np.allclose(M.Point @ M.Point.T, Copt @ Copt.T)

if __name__ == "__main__":
	Flag().testPrincipalComponentAnalysis()
