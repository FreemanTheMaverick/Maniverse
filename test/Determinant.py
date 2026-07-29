import unittest as ut
import numpy as np
import Maniverse as mv

# Alignment of two spaces
# Finding the maximal overlap of two spaces
# Minimize L(C) = det[ C0.t C ] ( -1 )
# C0, C \in Flag(5; 10) = Gr(5; 10)

class ObjDeterminant(mv.Objective):
	def __init__(self, C0):
		super().__init__()
		self.C0 = C0
		self.C = np.zeros([10, 5])
		self.C0tC = np.zeros([10, 5])
		self.C0tCinv = np.zeros([10, 5])
		self.rank = 0

		# Rank-deficient
		self.beta = 0

		# Rank-deficient 1
		self.u0 = np.zeros(10)
		self.v0 = np.zeros(10)

		# Rank-deficient 2
		self.U0 = np.zeros([10, 2])
		self.V0 = np.zeros([10, 2])

	def Calculate(self, C_, derivatives):
		self.C = C_[0]
		if 0 in derivatives:
			self.C0tC = self.C0.T @ self.C
			self.Value = np.linalg.det(self.C0tC)
			if len(derivatives) == 1:
				return
		self.rank = np.linalg.matrix_rank(self.C0tC)
		U, S, Vh = np.linalg.svd(self.C0tC)
		S = S[:self.rank]
		V = Vh.T
		self.C0tCinv = V[:, :self.rank] @ np.diag(1./S) @ U[:, :self.rank].T
		if self.rank == 5:
			self.Gradient = [ self.Value * self.C0 @ self.C0tCinv.T ]
		else:
			sing_prod = np.prod(S)
			detUV = np.linalg.det( U @ V )
			self.beta = sing_prod * detUV
			if self.rank == 4:
				self.u0 = U[:, 4]
				self.v0 = V[:, 4]
				self.Gradient = [ sing_prod * self.C0 @ np.outer( self.v0, self.u0 ) ]
			elif self.rank == 3:
				self.U0 = U[:, 3:]
				self.V0 = V[:, 3:]
				self.Gradient = [ np.zeros([10, 5]) ]
			else:
				self.Gradient = [ np.zeros([10, 5]) ]

	def Hessian(self, X_):
		X = X_[0]
		if self.rank == 5:
			return [ self.Value * self.C0 @ (
				np.sum( self.C0tCinv.T * ( self.C0.T @ X ) ) * self.C0tCinv.T
				- self.C0tCinv.T @ X.T @ self.C0 @ self.C0tCinv.T
			) ]
		if self.rank == 4:
			return [ 2 * self.beta * (
				self.C0 @ np.outer(self.u0, self.v0) * np.sum( self.C0tCinv * ( self.C0.T @ X ) )
				- self.C0 @ np.outer(self.u0, self.v0) @ X.T @ self.C0 @ self.C0tCinv.T
				- np.outer(self.u0, self.v0) @ self.C0 @ X @ self.C0 @ self.C0tCinv.T
			) ]
		if self.rank == 3:
			M = self.U0.T @ self.C0.T @ X @ self.V0
			M[0, 1] *= -1
			M[1, 0] *= -1
			M[0, 0], M[1, 1] = M[1, 1], M[0, 0]
			return [ 2 * self.beta * self.C0 @ self.U0 @ M.T @ self.V0.T ]
		return [ np.zeros([10, 5]) ]

class ObjDeterminants(mv.Objective):
	def __init__(self, C0s):
		super().__init__()
		self.Funcs = []
		for C0 in C0s:
			self.Funcs.append(ObjDeterminant(C0))

	def Calculate(self, Cs_, derivatives):
		self.Value = 0
		gradient = np.zeros_like(Cs_[0])
		for func in self.Funcs:
			func.Calculate(Cs_, derivatives)
			self.Value += func.Value
			gradient += func.Gradient[0]
		self.Gradient = [ gradient ]

	def Hessian(self, Xs_):
		HX = np.zeros_like(Xs_[0])
		for func in self.Funcs:
			HX += func.Hessian(Xs_)[0]
		return [ HX ]

class TestDeterminant(ut.TestCase):
	def __init__(self, *args):
		super().__init__(*args)
		_, eigvecs = np.linalg.eigh(np.loadtxt("Sym10.txt", delimiter = ',').reshape([10, 10]))
		self.Obj = ObjDeterminant(eigvecs[:, :5])
		self.Manifold = mv.Flag(np.eye(10, 5))
		self.Manifold.setBlockParameters([5])
		self.Tolerance = (1.e-5, 1.e-5, 1.e-5)
		self.Solution = - eigvecs[:, :5]

	def testNewtonCG(self):
		M = mv.Iterate(self.Obj, [self.Manifold])
		tr = mv.TrustRegion()
		cg = mv.ConjugateGradient(M, 0, 1, (1e-4, 1e-4), M.getDimension(), 0)
		converged = mv.Newton(
				M, tr, cg, self.Tolerance, 8, 0
		)
		assert converged
		assert np.allclose(M.Ms[0].P @ M.Ms[0].P.T, self.Solution @ self.Solution.T, atol = 1e-5)

	def testLBFGS(self):
		M = mv.Iterate(self.Obj, [self.Manifold])
		converged = mv.LBFGS(
				M, self.Tolerance,
				10, 10, 0.1, 0.75, 15, 0
		)
		assert converged
		assert np.allclose(M.Ms[0].P @ M.Ms[0].P.T, self.Solution @ self.Solution.T, atol = 1e-5)

	def testLanczos(self):
		M = mv.Iterate(self.Obj, [self.Manifold])
		M.setPoint([self.Solution], 1)
		M.Func.Calculate(M.getPoint(), [0, 1, 2])
		M.setGradient()
		Evals, Evecs = mv.Lanczos(M, 1, 0, 0, 0)
		residual = np.linalg.norm( M.Hessian(Evecs[0]) - Evals[0] * Evecs[0] )
		assert residual < 1e-5

if __name__ == "__main__":
	TestDeterminant().testNewtonCG()
	TestDeterminant().testLBFGS()
	TestDeterminant().testLanczos()
