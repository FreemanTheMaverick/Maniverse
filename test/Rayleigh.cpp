#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <iostream>
#include <Maniverse/Manifold/Stiefel.h>
#include <Maniverse/LinearSolver/ConjugateGradient.h>
#include <Maniverse/Optimizer/Newton.h>
#include <Maniverse/Optimizer/LBFGS.h>
#include <Maniverse/Diagonalizer/Lanczos.h>

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

	void Calculate(std::vector<Eigen::MatrixXd> C, std::vector<int> derivatives) override{
		if ( std::count(derivatives.begin(), derivatives.end(), 0) ){
			Value = C[0].cwiseProduct( A * C[0] ).sum();
		}
		if ( std::count(derivatives.begin(), derivatives.end(), 1) ){
			Gradient = { 2 * A * C[0] };
		}
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

#define __Check_Stability__\
	std::cout << typeid(*this).name() << " " << __func__ << " ";\
	for ( int i = 0; i < (int)Evecs.size(); i++ ){\
		const double residual = ( M.ConstraintProjectedHessian(Evecs[i]) - Evals[i] * Evecs[i] ).norm();\
		if ( residual > 1e-5 ) goto IncorrectCurvature;\
	}\
	std::cout << "\033[32mSuccess!\033[0m" << std::endl; return;\
	IncorrectCurvature: std::cout << "\033[31mFailed: Eigenvalue equation is violated!\033[0m" << std::endl;

class TestRayleigh{ public:
	ObjRayleigh Obj = ObjRayleigh();
	mv::Stiefel Manifold = mv::Stiefel(Eigen::MatrixXd::Identity(10, 1));
	std::tuple<double, double, double> Tolerance = {1.e-5, 1.e-5, 1.e-5};
	Eigen::MatrixXd Solution = Eigen::MatrixXd::Zero(10, 1);

	TestRayleigh(){
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
		es.compute(Obj.A);
		const Eigen::MatrixXd Evec = es.eigenvectors();
		Manifold = mv::Stiefel( ( Evec.col(0) + Evec.col(1) ) / std::sqrt(2) );
		Solution = Evec.col(0);
	};

	void testNewtonCG(){
		mv::Iterate M(Obj, {Manifold.Share()});
		mv::TrustRegion tr;
		mv::ConjugateGradient cg(M, 0, 1, {1e-4, 1e-4}, M.getDimension(), 1);
		const bool converged = mv::Newton(
				M, tr, cg, Tolerance, 3, 1
		);
		__Check_Result__
	};

	void testLBFGS(){
		mv::Iterate M(Obj, {Manifold.Share()});
		const bool converged = mv::LBFGS(
				M, Tolerance,
				10, 8, 0.1, 0.75, 5, 1
		);
		__Check_Result__
	};

	void testLanczos(){
		mv::Iterate M(Obj, {Manifold.Share()});
		M.setPoint({Solution}, 1);
		M.Func->Calculate(M.getPoint(), {0, 1, 2});
		M.setGradient();
		const auto [Evals, Evecs] = mv::Lanczos(M, M.getDimension(), 0, 0, 1);
		__Check_Stability__
	};
};

int main(){
	TestRayleigh().testNewtonCG();
	TestRayleigh().testLBFGS();
	TestRayleigh().testLanczos();
}
