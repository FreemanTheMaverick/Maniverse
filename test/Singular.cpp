#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <Maniverse/Manifold/Stiefel.h>
#include <Maniverse/Manifold/Euclidean.h>
#include <Maniverse/Manifold/Orthogonal.h>
#include <Maniverse/LinearSolver/ConjugateGradient.h>
#include <Maniverse/Optimizer/TruncatedNewton.h>
#include <Maniverse/Optimizer/LBFGS.h>
#include <Maniverse/Diagonalizer/Lanczos.h>

// Thin singular value decomposition
// Finding the singular values and vectors of a rectangular A
// Minimize L(U, s, V) = || U diag(s) V.t - A ||^2
// A \in R(10, 6)
// U \in St(10, 6)
// s \in R(6)
// V \in O(6)

namespace mv = Maniverse;

class ObjSingular: public mv::Objective{ public:
	Eigen::MatrixXd A = Eigen::MatrixXd::Zero(10, 6);
	Eigen::MatrixXd U = Eigen::MatrixXd::Zero(10, 6);
	Eigen::MatrixXd s = Eigen::MatrixXd::Zero(6, 1);
	Eigen::MatrixXd V = Eigen::MatrixXd::Zero(6, 6);

	ObjSingular(){
		const double data[] = {
			#include "Sym10.txt"
		};
		std::memcpy(A.data(), &data, 10 * 6 * 8);
	};

	void Calculate(std::vector<Eigen::MatrixXd> X, std::vector<int> derivatives) override{
		if ( std::count(derivatives.begin(), derivatives.end(), 0) ){
			U = X[0];
			s = X[1];
			V = X[2];
			Value = ( U * s.asDiagonal() * V.transpose() - A ).squaredNorm();
		}
		if ( std::count(derivatives.begin(), derivatives.end(), 1) ){
			const Eigen::MatrixXd GU = 2 * ( U * s.asDiagonal() * s.asDiagonal() - A * V * s.asDiagonal() );
			const Eigen::MatrixXd Gs = 2 * ( s - ( U.transpose() * A * V ).diagonal() );
			const Eigen::MatrixXd GV = 2 * ( V * s.asDiagonal() * s.asDiagonal() - A.transpose() * U * s.asDiagonal() );
			Gradient = { GU, Gs, GV };
		}
	};

	std::vector<Eigen::MatrixXd> Hessian(std::vector<Eigen::MatrixXd> K) const override{
		const Eigen::MatrixXd& delta_U = K[0];
		const Eigen::MatrixXd& delta_s = K[1];
		const Eigen::MatrixXd& delta_V = K[2];
		const Eigen::MatrixXd HUU = 2 * delta_U * s.asDiagonal() * s.asDiagonal();
		const Eigen::MatrixXd HUs = 4 * U * s.asDiagonal() * delta_s.asDiagonal() - 2 * A * V * delta_s.asDiagonal();
		const Eigen::MatrixXd HUV = - 2 * A * delta_V * s.asDiagonal();
		const Eigen::MatrixXd HsU = - 2 * ( delta_U.transpose() * A * V ).diagonal();
		const Eigen::MatrixXd Hss = 2 * delta_s;
		const Eigen::MatrixXd HsV = - 2 * ( U.transpose() * A * delta_V ).diagonal();
		const Eigen::MatrixXd HVU = - 2 * A.transpose() * delta_U * s.asDiagonal();
		const Eigen::MatrixXd HVs = 4 * V * s.asDiagonal() * delta_s.asDiagonal() - 2 * A.transpose() * U * delta_s.asDiagonal();
		const Eigen::MatrixXd HVV = 2 * delta_V * s.asDiagonal() * s.asDiagonal();
		return std::vector<Eigen::MatrixXd>{
			HUU + HUs + HUV,
			HsU + Hss + HsV,
			HVU + HVs + HVV
		};
	};
};

#define __Check_Result__\
	std::cout << typeid(*this).name() << " " << __func__ << " ";\
	if ( converged ){\
		if ( ( M.Ms[0]->P * M.Ms[1]->P.asDiagonal() * M.Ms[2]->P.transpose() - Obj.A ).cwiseAbs().maxCoeff() < 1e-5 ){\
			std::cout << "\033[32mSuccess!\033[0m" << std::endl;\
		}else std::cout << "\033[31mFailed: Incorrect solution!\033[0m" << std::endl;\
	}else std::cout << "\033[31mFailed: Not converged!\033[0m" << std::endl;

#define __Check_Stability__\
	std::cout << typeid(*this).name() << " " << __func__ << " ";\
	for ( int i = 0; i < (int)Evecs.size(); i++ ){\
		const double residual = ( M.ConstraintProjectedHessian(Evecs[i]) - Evals[i] * Evecs[i] ).norm();\
		if ( residual > 1e-5 ) goto IncorrectCurvature;\
	}\
	std::cout << "\033[32mSuccess!\033[0m" << std::endl; return;\
	IncorrectCurvature: std::cout << "\033[31mFailed: Eigenvalue equation is violated!\033[0m" << std::endl;

class TestSingular{ public:
	ObjSingular Obj = ObjSingular();
	mv::Stiefel Manifold0 = mv::Stiefel(Eigen::MatrixXd::Identity(10, 6));
	mv::Euclidean Manifold1 = mv::Euclidean(Eigen::MatrixXd::Zero(6, 1));
	mv::Orthogonal Manifold2 = mv::Orthogonal(Eigen::MatrixXd::Identity(6, 6));
	std::tuple<double, double, double> Tolerance = {1.e-5, 1.e-5, 1.e-5};
	Eigen::MatrixXd Solution0 = Eigen::MatrixXd::Identity(10, 6);
	Eigen::MatrixXd Solution1 = Eigen::MatrixXd::Zero(6, 1);
	Eigen::MatrixXd Solution2 = Eigen::MatrixXd::Identity(6, 6);

	TestSingular(){
		Eigen::BDCSVD<Eigen::MatrixXd, Eigen::ComputeThinU | Eigen::ComputeFullV> svd(Obj.A);
		Solution0 = svd.matrixU();
		Solution1 = svd.singularValues();
		Solution2 = svd.matrixV();
	}

	void testTruncatedNewton(){
		mv::TrustRegion tr;
		mv::ConjugateGradient cg(3, 1, {1e-4, 1e-4}, 1);
		mv::Iterate M(Obj, {Manifold0.Share(), Manifold1.Share(), Manifold2.Share()});
		const bool converged = mv::TruncatedNewton(
				M, tr, cg, Tolerance, 21, 1
		);
		__Check_Result__
	};

	void testLBFGS(){
		mv::Iterate M(Obj, {Manifold0.Share(), Manifold1.Share(), Manifold2.Share()});
		const bool converged = mv::LBFGS(
				M, Tolerance,
				100, 131, 0.1, 0.75, 5, 1
		);
		__Check_Result__
	};

	void testLanczos(){
		mv::Iterate M(Obj, {Manifold0.Share(), Manifold1.Share(), Manifold2.Share()});
		M.setPoint({Solution0, Solution1, Solution2}, 1);
		M.Func->Calculate(M.getPoint(), {0, 1, 2});
		M.setGradient();
		const auto [Evals, Evecs] = mv::Lanczos(M, M.getDimension(), 1);
		__Check_Stability__
	};
};

int main(){
	TestSingular().testTruncatedNewton();
	TestSingular().testLBFGS();
	TestSingular().testLanczos();
}
