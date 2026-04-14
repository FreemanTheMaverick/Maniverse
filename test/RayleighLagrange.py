import unittest as ut
import numpy as np
import Maniverse as mv

# Rayleigh quotient
# Finding the smallest eigenvalue of A
# Minimize L(C) = C.t A C
# A \in Sym(10)
# C \in St(10, 1)

class Obj(mv.Objective):
	def __init__(self):
		super().__init__()
		self.A = np.loadtxt("Sym10.txt", delimiter = ',').reshape([10, 10])
		Eval, Evec = np.linalg.eigh(self.A)
		self.A = Evec * np.abs(Eval) @ Evec.T
		self.C = np.zeros([10, 1])
		self.Cnorm2 = 0
		self.Lambda = [0]

	def Calculate(self, C_, derivatives):
		C = self.C = C_[0]
		Cnorm2 = self.Cnorm2 = np.linalg.norm(C) ** 2
		if 0 in derivatives:
			self.Value = (
					np.sum( C * ( self.A @ C ) )
					+ self.Lambda[0] * ( Cnorm2 - 1 )
					+ self.Rho / 2 * ( Cnorm2 - 1 ) ** 2
			)
			self.Constraint_Value = [ Cnorm2 - 1 ]
		if 1 in derivatives:
			self.Gradient = [
					2 * self.A @ C
					+ self.Lambda[0] * 2 * C
					+ self.Rho * ( Cnorm2 - 1 ) * 2 * C
			]
			self.Constraint_Gradient = [[ 2 * C ]]

	def Hessian(self, V_):
		V = V_[0]
		return [
				2 * self.A @ V
				+ self.Lambda[0] * 2 * V
				+ self.Rho * ( self.Cnorm2 - 1 ) * 2 * V
				+ self.Rho * 2 * np.sum( self.C * V ) * 2 * self.C
		]

class TestRayleigh(ut.TestCase):
	def __init__(self, *args):
		super().__init__(*args)
		self.Obj = Obj()
		_, Evec = np.linalg.eigh(self.Obj.A)
		self.Manifold = mv.Euclidean( ( Evec[:, 0] + Evec[:, 1] ) / np.sqrt(2) )
		self.Tolerance = (1.e-5, 1.e-5, 1.e-5)
		self.TrustRegion = mv.TrustRegion()
		self.Solution = Evec[:, 0]

	def testTruncatedNewton(self):
		M = mv.Iterate(self.Obj, {self.Manifold}, True)
		converged = mv.AugmentedLagrangian(1, 3.3, 0.8, {1e-5}, 4, 0)(mv.TruncatedNewton)(
				M, self.TrustRegion, self.Tolerance,
				0.001, 10, 0
		)
		assert converged
		assert np.allclose(M.Ms[0].P[:, 0], self.Solution, atol = 1e-5)

	def testLBFGS(self):
		M = mv.Iterate(self.Obj, {self.Manifold}, True)
		converged = mv.AugmentedLagrangian(1, 3.3, 0.8, {1e-5}, 4, 0)(mv.LBFGS)(
				M, self.Tolerance,
				10, 20, 0.1, 0.75, 7, 0
		)
		assert converged
		assert np.allclose(M.Ms[0].P[:, 0], self.Solution, atol = 1e-5)

if __name__ == "__main__":
	TestRayleigh().testTruncatedNewton()
	TestRayleigh().testLBFGS()
