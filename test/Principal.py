import unittest as ut
import numpy as np
import Maniverse as mv

# Principal component analysis
# Finding the space spanned by the highest 5 eigenvectors
# Maximize L(C) = Tr[ C.t A C ]
# A \in Sym(10)
# C \in Flag(5; 10) = Gr(5; 10)

class Obj(mv.Objective):
	def __init__(self):
		super().__init__()
		self.A = np.loadtxt("Sym10.txt", delimiter = ',').reshape([10, 10])
	
	def Calculate(self, C, derivatives):
		if 0 in derivatives:
			self.Value = - np.sum( C[0] * ( self.A @ C[0] ) )
		if 1 in derivatives:
			self.Gradient = [ - 2 * self.A @ C[0] ]

	def Hessian(self, V):
		return [ - 2 * self.A @ V[0] ]

class TestPrincipal(ut.TestCase):
	def __init__(self, *args):
		super().__init__(*args)
		self.Obj = Obj()
		self.Manifold = mv.Flag(np.eye(10)[:, :5]) # Initial guess
		self.Manifold.setBlockParameters([5])
		self.Tolerance = (1.e-5, 1.e-5, 1.e-5)
		self.Solution = np.linalg.eigh(self.Obj.A)[1][:, 5:]

	def testNewtonCG(self):
		M = mv.Iterate(self.Obj, [self.Manifold])
		tr = mv.TrustRegion()
		cg = mv.ConjugateGradient(M, 0, 1, (1e-4, 1e-4), M.getDimension(), 0)
		converged = mv.Newton(
				M, tr, cg, self.Tolerance, 8, 0
		)
		assert converged
		assert np.allclose(M.Ms[0].P @ M.Ms[0].P.T, self.Solution @ self.Solution.T, atol = 1e-5)

	def testNewtonMR(self):
		M = mv.Iterate(self.Obj, [self.Manifold])
		tr = mv.TrustRegion()
		mr = mv.MinRes(M, 0, 1, (1e-4, 1e-4), M.getDimension(), 0)
		converged = mv.Newton(
				M, tr, mr, self.Tolerance, 10, 0
		)
		assert converged
		assert np.allclose(M.Ms[0].P @ M.Ms[0].P.T, self.Solution @ self.Solution.T, atol = 1e-5)

	def testLBFGS(self):
		M = mv.Iterate(self.Obj, [self.Manifold])
		converged = mv.LBFGS(
				M, self.Tolerance,
				10, 43, 0.1, 0.75, 5, 0
		)
		assert converged
		assert np.allclose(M.Ms[0].P @ M.Ms[0].P.T, self.Solution @ self.Solution.T, atol = 1e-5)

	def testLanczos(self):
		M = mv.Iterate(self.Obj, [self.Manifold])
		M.setPoint([self.Solution], 1)
		M.Func.Calculate(M.getPoint(), [0, 1, 2])
		M.setGradient()
		Evals, Evecs = mv.Lanczos(M, M.getDimension(), 0, 0, 0)
		for i in range(len(Evecs)):
			residual = np.linalg.norm( M.ConstraintProjectedHessian(Evecs[i]) - Evals[i] * Evecs[i] )
			assert residual < 1e-5

if __name__ == "__main__":
	TestPrincipal().testNewtonCG()
	TestPrincipal().testNewtonMR()
	TestPrincipal().testLBFGS()
	TestPrincipal().testLanczos()
