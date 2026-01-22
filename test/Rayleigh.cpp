#include <Eigen/Dense>
#include <cmath>
#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <Maniverse/Manifold/Stiefel.h>
#include <Maniverse/Optimizer/TruncatedNewton.h>
#include <Maniverse/Optimizer/LBFGS.h>

// Rayleigh quotient
// Finding the smallest eigenvalue of A
// Minimize L(C) = C.t A C
// A \in Sym(10)
// C \in St(10, 1)

namespace mv = Maniverse;

class ObjRayleigh: public mv::Objective{ public:
	Eigen::MatrixXd A = Eigen::MatrixXd::Zero(10, 10);

	ObjRayleigh(){
		const double data[] = {
			#include "Sym10.txt"
		};
		std::memcpy(A.data(), &data, 10 * 10 * 8);
	};

	void Calculate(std::vector<Eigen::MatrixXd> C, int /*derivative*/) override{
		Value = C[0].cwiseProduct( A * C[0] ).sum();
		Gradient = { 2 * A * C[0] };
	};

	std::vector<Eigen::MatrixXd> Hessian(std::vector<Eigen::MatrixXd> V) const override{
		return std::vector<Eigen::MatrixXd>{ 2 * A * V[0] };
	};
};

#define __Check_Result__\
	std::cout << typeid(*this).name() << " " << __func__ << " ";\
	if ( converged ){\
		if ( ( M.Ms[0]->P - Solution ).cwiseAbs().maxCoeff() < 1e-5 ){\
			std::cout << "\033[32mSuccess!\033[0m" << std::endl;\
		}else std::cout << "\033[31mFailed: Incorrect solution!\033[0m" << std::endl;\
	}else std::cout << "\033[31mFailed: Not converged!\033[0m" << std::endl;

class TestRayleigh{ public:
	ObjRayleigh Obj = ObjRayleigh();
	mv::Stiefel Manifold = mv::Stiefel(Eigen::MatrixXd::Identity(10, 1));
	std::tuple<double, double, double> Tolerance = {1.e-5, 1.e-5, 1.e-5};
	mv::TrustRegion TrustRegion = mv::TrustRegion();
	Eigen::MatrixXd Solution = Eigen::MatrixXd::Zero(10, 1);

	TestRayleigh(){
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
		es.compute(Obj.A);
		const Eigen::MatrixXd Evec = es.eigenvectors();
		Manifold = mv::Stiefel( ( Evec.col(0) + Evec.col(1) ) / std::sqrt(2) );
		Solution = Evec.col(0);
	};

	void testTruncatedNewton(){
		mv::Iterate M(Obj, {Manifold.Share()}, true);
		const bool converged = mv::TruncatedNewton(
				M, TrustRegion, Tolerance,
				0.001, 3, 1
		);
		__Check_Result__
	};

	void testLBFGS(){
		mv::Iterate M(Obj, {Manifold.Share()}, true);
		const bool converged = mv::LBFGS(
				M, Tolerance,
				10, 8, 0.1, 0.75, 5, 1
		);
		__Check_Result__
	};
};

int main(){
	TestRayleigh().testTruncatedNewton();
	TestRayleigh().testLBFGS();
}
