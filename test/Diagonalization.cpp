#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <iostream>
#include <Maniverse/Manifold/Euclidean.h>
#include <Maniverse/Manifold/Orthogonal.h>
#include <Maniverse/LinearSolver/ConjugateGradient.h>
#include <Maniverse/Optimizer/Newton.h>
#include <Maniverse/Optimizer/LBFGS.h>
#include <Maniverse/Diagonalizer/Lanczos.h>
#include <Maniverse/LinearSolver/MinRes.h>
//#include <Maniverse/Diagonalizer/Davidson.h>

// Symmetric diagonalization
// Finding the eigenvalues and eigenvectors of a symmetric A
// Minimize L(n, C) = || C diag(n) C.t - A ||^2
// A \in Sym(10)
// n \in R(10)
// C \in O(10)

namespace mv = Maniverse;

class ObjDiagonalization: public mv::Objective{ public:
	Eigen::MatrixXd A = Eigen::MatrixXd::Zero(10, 10);
	Eigen::MatrixXd n = Eigen::MatrixXd::Zero(10, 1);
	Eigen::MatrixXd C = Eigen::MatrixXd::Zero(10, 10);

	ObjDiagonalization(){
		const double data[] = {
			#include "Sym10.txt"
		};
		std::memcpy(A.data(), &data, 10 * 10 * 8);
	};

	void Calculate(std::vector<Eigen::MatrixXd> X, std::vector<int> derivatives) override{
		if ( std::count(derivatives.begin(), derivatives.end(), 0) ){
			n = X[0];
			C = X[1];
			Value = std::pow(( C * n.asDiagonal() * C.transpose() - A ).norm(), 2);
		}
		if ( std::count(derivatives.begin(), derivatives.end(), 1) ){
			const Eigen::MatrixXd Gn = 2 * ( n - ( C.transpose() * A * C ).diagonal() );
			const Eigen::MatrixXd GC = 4 * ( C * n.asDiagonal() * n.asDiagonal() - A * C * n.asDiagonal() );
			Gradient = { Gn, GC };
		}
	};

	std::vector<Eigen::MatrixXd> Hessian(std::vector<Eigen::MatrixXd> V) const override{
		const Eigen::MatrixXd& delta_n = V[0];
		const Eigen::MatrixXd& delta_C = V[1];
		const Eigen::MatrixXd Hnn = 2 * delta_n;
		const Eigen::MatrixXd HnC = - 4 * ( C.transpose() * A * delta_C ).diagonal();
		const Eigen::MatrixXd HCn = 8 * C * n.asDiagonal() * delta_n.asDiagonal() - 4 * A * C * delta_n.asDiagonal();
		const Eigen::MatrixXd HCC = 4 * ( delta_C * n.asDiagonal() * n.asDiagonal() - A * delta_C * n.asDiagonal() );
		return std::vector<Eigen::MatrixXd>{
			Hnn + HnC,
			HCn + HCC
		};
	};
};

#define __Check_Result__\
	std::cout << typeid(*this).name() << " " << __func__ << " ";\
	if ( converged ){\
		if ( ( M.Ms[1]->P * M.Ms[0]->P.asDiagonal() * M.Ms[1]->P.transpose() - Obj.A ).cwiseAbs().maxCoeff() < 1e-5 ){\
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

class TestDiagonalization{ public:
	ObjDiagonalization Obj = ObjDiagonalization();
	mv::Euclidean Manifold0 = mv::Euclidean(Eigen::MatrixXd::Zero(10, 1));
	mv::Orthogonal Manifold1 = mv::Orthogonal(Eigen::MatrixXd::Identity(10, 10));
	std::tuple<double, double, double> Tolerance = {1.e-5, 1.e-5, 1.e-5};
	Eigen::MatrixXd Solution0 = Eigen::MatrixXd::Zero(10, 1);
	Eigen::MatrixXd Solution1 = Eigen::MatrixXd::Zero(10, 10);

	TestDiagonalization(){
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Obj.A);
		Solution0 = es.eigenvalues();
		Solution1 = es.eigenvectors();
	};

	void testNewtonCG(){
		mv::Iterate M(Obj, {Manifold0.Share(), Manifold1.Share()});
		mv::TrustRegion tr;
		mv::ConjugateGradient cg(M, 0, 1, {1e-4, 1e-4}, M.getDimension(), 1);
		const bool converged = mv::Newton(
				M, tr, cg, Tolerance, 26, 1
		);
		__Check_Result__
	};

	void testLBFGS(){
		mv::Iterate M(Obj, {Manifold0.Share(), Manifold1.Share()});
		const bool converged = mv::LBFGS(
				M, Tolerance,
				100, 110, 0.1, 0.75, 5, 1
		);
		__Check_Result__
	};

	void testLanczos(){
		mv::Iterate M(Obj, {Manifold0.Share(), Manifold1.Share()});
		M.setPoint({Solution0, Solution1}, 1);
		M.Func->Calculate(M.getPoint(), {0, 1, 2});
		M.setGradient();
		const auto [Evals, Evecs] = mv::Lanczos(M, M.getDimension(), 0, 0, 1);
		__Check_Stability__
	};

	/*
	void testDavidson(){
		mv::Iterate M(Obj, {Manifold0.Share(), Manifold1.Share()});
		M.setPoint({Solution0, Solution1}, 1);
		M.Func->Calculate(M.getPoint(), {0, 1, 2});
		M.setGradient();
		mv::MinRes mr(1, {1e-12, 1e-12}, 1);
		mr.M = &M;
		const auto [Evals, Evecs] = mv::Davidson(M, mr, 0, -114514, 1e-8, 100, 1, 0, 0, 1);
		__Check_Stability__
	};*/
};

int main(){
	TestDiagonalization().testNewtonCG();
	TestDiagonalization().testLBFGS();
	TestDiagonalization().testLanczos();
	//TestDiagonalization().testDavidson();
}
