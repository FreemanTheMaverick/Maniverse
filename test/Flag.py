import unittest as ut
import numpy as np
import Maniverse as mv

# L(C) = Tr[ C.t A C ]
# A \in Sym(10)
# C \in Flag(1, 3, 6; 10)

class Flag(ut.TestCase):
	def test(self):
		A = np.fromfile("Flag_A.dat", dtype=np.float64)
		A.shape = (10, 10)
		def Objective(Cs, _):
			C = Cs[0]
			L = np.sum( C * ( A @ C ) )
			G = 2 * A @ C
			def H(v):
				return 2 * A @ v
			return L, [G], [H]
		flag = mv.Flag(np.eye(10)[:, :6]);
		flag.setBlockParameters({1, 2, 3});
		M = mv.Iterate({flag.Clone()}, True)
		L = 0
		tr_setting = mv.TrustRegionSetting()
		tol = (1.e-5, 1.e-5, 1.e-5) 
		conv = mv.TrustRegion(
				Objective, tr_setting, tol,
				0.001, 1, 1000, L, M, 1
		)
		assert conv == 1
		assert L == 123

if __name__ == "__main__":
	Flag().test()
