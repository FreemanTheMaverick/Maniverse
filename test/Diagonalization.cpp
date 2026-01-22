#include <Eigen/Dense>
#include <cmath>
#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <Maniverse/Manifold/Euclidean.h>
#include <Maniverse/Manifold/Orthogonal.h>
#include <Maniverse/Optimizer/TruncatedNewton.h>
#include <Maniverse/Optimizer/LBFGS.h>

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

	void Calculate(std::vector<Eigen::MatrixXd> X, int /*derivative*/) override{
		n = X[0];
		C = X[1];
		Value = std::pow(( C * n.asDiagonal() * C.transpose() - A ).norm(), 2);
		const Eigen::MatrixXd Gn = 2 * ( n - ( C.transpose() * A * C ).diagonal() );
		const Eigen::MatrixXd GC = 4 * ( C * n.asDiagonal() * n.asDiagonal() - A * C * n.asDiagonal() );
		Gradient = { Gn, GC };
	};

	std::vector<std::vector<Eigen::MatrixXd>> Hessian(std::vector<Eigen::MatrixXd> V) const override{
		const Eigen::MatrixXd& delta_n = V[0];
		const Eigen::MatrixXd& delta_C = V[1];
		const Eigen::MatrixXd Hnn = 2 * delta_n;
		const Eigen::MatrixXd HnC = - 4 * ( C.transpose() * A * delta_C ).diagonal();
		const Eigen::MatrixXd HCn = 8 * C * n.asDiagonal() * delta_n.asDiagonal() - 4 * A * C * delta_n.asDiagonal();
		const Eigen::MatrixXd HCC = 4 * ( delta_C * n.asDiagonal() * n.asDiagonal() - A * delta_C * n.asDiagonal() );
		return std::vector<std::vector<Eigen::MatrixXd>>{
			{ Hnn, HnC },
			{ HCn, HCC }
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

class TestDiagonalization{ public:
	ObjDiagonalization Obj = ObjDiagonalization();
	mv::Euclidean Manifold0 = mv::Euclidean(Eigen::MatrixXd::Zero(10, 1));
	mv::Orthogonal Manifold1 = mv::Orthogonal(Eigen::MatrixXd::Identity(10, 10));
	std::tuple<double, double, double> Tolerance = {1.e-5, 1.e-5, 1.e-5};
	mv::TrustRegion TrustRegion = mv::TrustRegion();

	void testTruncatedNewton(){
		mv::Iterate M(Obj, {Manifold0.Share(), Manifold1.Share()}, true);
		const bool converged = mv::TruncatedNewton(
				M, TrustRegion, Tolerance,
				0.0001, 26, 1
		);
		__Check_Result__
	};

	void testLBFGS(){
		mv::Iterate M(Obj, {Manifold0.Share(), Manifold1.Share()}, true);
		const bool converged = mv::LBFGS(
				M, Tolerance,
				100, 110, 0.1, 0.75, 5, 1
		);
		__Check_Result__
	};
};

int main(){
	TestDiagonalization().testTruncatedNewton();
	TestDiagonalization().testLBFGS();
}
