import unittest as ut
import numpy as np
import Maniverse as mv

class Euclidean(ut.TestCase):

	def testQuadratic(self):
		# Quadratic minimization
		# Finding the bottom of a quadratic form
		# Minimize L(x) = x.t A x
		# A \in SPD(10), nearly diagonal
		# x \in R(10)
		A = np.fromfile("Sym10.dat").reshape([10, 10])
		A = A @ A + np.eye(10) * 0.01 # Constructing a SPD matrix
		for i in range(10):
			for j in range(10):
				if i != j:
					A[i, j] *= 0.01

		x0 = range(10)
		def Objective(xs, _):
			x = xs[0]
			L = np.sum( x * ( A @ x ) )
			G = 2 * A @ x
			return L, [G]
		euclidean = mv.Euclidean(x0)
		M = mv.Iterate({euclidean.Clone()}, True)
		L = 0
		tol = (1.e-5, 1.e-5, 1.e-5) 
		converged = mv.LBFGS(
				Objective, tol,
				20, 12, L, M, 0
		)
		assert converged
		assert np.allclose(M.Point, np.zeros_like(M.Point), atol = 1e-5)

	def testPreconditionedQuadratic(self):
		# Preconditioned quadratic minimization
		# Finding the bottom of a quadratic form
		# Minimize L(x) = x.t A x
		# A \in SPD(10), nearly diagonal
		# x \in R(10)
		A = np.fromfile("Sym10.dat").reshape([10, 10])
		A = A @ A + np.eye(10) * 0.01 # Constructing a SPD matrix

		for i in range(10):
			for j in range(10):
				if i != j:
					A[i, j] *= 0.01
		Asqrt = np.diag( np.sqrt( np.abs( np.diag( A ) ) ) )
		Ainvsqrt = np.linalg.inv(Asqrt)

		x0 = range(10)
		def Objective(xs, _):
			x = xs[0]
			L = np.sum( x * ( A @ x ) )
			G = 2 * A @ x
			def P(a):
				return Ainvsqrt @ a
			def invP(a):
				return Asqrt @ a
			return L, [G], [P], [invP]
		euclidean = mv.Euclidean(x0)
		M = mv.Iterate({euclidean.Clone()}, True)
		L = 0
		tol = (1.e-5, 1.e-5, 1.e-5) 
		converged = mv.PreconLBFGS(
				Objective, tol,
				20, 8, L, M, 0
		)
		assert converged
		assert np.allclose(M.Point, np.zeros_like(M.Point), atol = 1e-5)

if __name__ == "__main__":
	Euclidean().testQuadratic()
	Euclidean().testPreconditionedQuadratic()
