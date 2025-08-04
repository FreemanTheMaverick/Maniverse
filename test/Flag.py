import unittest as ut
import numpy as np
import Maniverse as mv

# L(C) = Tr[ C.t A C ]
# A \in Sym(10)
# C \in Stiefel(10, 6)

class Stiefel(ut.TestCase):
	def test(self):
		A = np.fromfile("Symmetric.dat", dtype = np.float64)
		A.shape = (10, 10)
		def Objective(Cs, _):
			C = Cs[0]
			L = - np.sum( C * ( A @ C ) )
			G = - 2 * A @ C
			def H(v):
				return - 2 * A @ v
			return L, [G], [H]
		C0 = np.fromfile("Stiefel_guess.dat", dtype = np.float64)
		C0.shape = (10, 6)
		flag = mv.flag(C0)
		M = mv.Iterate({flag.Clone()}, True)
		L = 0
		tr_setting = mv.TrustRegionSetting()
		tol = (1.e-5, 1.e-5, 1.e-5) 
		conv = mv.TrustRegion(
				Objective, tr_setting, tol,
				0.001, 1, 10, L, M, 1
		)
		assert conv == 1
		assert L == 123

if __name__ == "__main__":
	Stiefel().test()
