#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <iostream>
#include <Maniverse/Manifold/Euclidean.h>
#include <Maniverse/Optimizer/AugmentedLagrangian.h>
#include <Maniverse/LinearSolver/ConjugateGradient.h>
#include <Maniverse/Optimizer/TruncatedNewton.h>
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
	Eigen::MatrixXd C = Eigen::MatrixXd::Zero(10, 1);
	double Cnorm2 = 0;

	ObjRayleigh(){
		const double data[] = {
			#include "Sym10.txt"
		};
		std::memcpy(A.data(), &data, 10 * 10 * 8);
		Lambda.resize(1);
	};

	void Calculate(std::vector<Eigen::MatrixXd> C_, std::vector<int> derivatives) override{
		C = C_[0];
		Cnorm2 = C.norm() * C.norm();
		if ( std::count(derivatives.begin(), derivatives.end(), 0) ){
			Value =
				C.cwiseProduct( A * C ).sum()
				+ Lambda[0] * ( Cnorm2 - 1 )
				+ Rho / 2 * ( Cnorm2 - 1 ) * ( Cnorm2 - 1 );
			Constraint_Value = { Cnorm2 - 1 };
		}
		if ( std::count(derivatives.begin(), derivatives.end(), 1) ){
			Gradient = {
				2 * A * C
				+ Lambda[0] * 2 * C
				+ Rho * ( Cnorm2 - 1 ) * 2 * C
			};
			Constraint_Gradient = {{ 2 * C }};
		}
	};

	std::vector<Eigen::MatrixXd> Hessian(std::vector<Eigen::MatrixXd> V_) const override{
		const Eigen::MatrixXd& V = V_[0];
		return std::vector<Eigen::MatrixXd>{
			2 * A * V
			+ Lambda[0] * 2 * V
			+ Rho * ( Cnorm2 - 1 ) * 2 * V
			+ Rho * 2 * C.cwiseProduct(V).sum() * 2 * C
		};
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

class TestRayleighLagrange{ public:
	ObjRayleigh Obj = ObjRayleigh();
	mv::Euclidean Manifold = mv::Euclidean(Eigen::MatrixXd::Identity(10, 1));
	std::tuple<double, double, double> Tolerance = {1.e-5, 1.e-5, 1.e-5};
	Eigen::MatrixXd Solution = Eigen::MatrixXd::Zero(10, 1);
	double Lambda = 0;

	TestRayleighLagrange(){
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
		es.compute(Obj.A);
		const Eigen::MatrixXd Evec = es.eigenvectors();
		Manifold = mv::Euclidean( ( Evec.col(0) + Evec.col(1) ) / std::sqrt(2) );
		Solution = Evec.col(0);
		Lambda = - es.eigenvalues()(0);
	};

	void testTruncatedNewton(){
		mv::TrustRegion tr;
		mv::ConjugateGradient cg(3, 1, {1e-4, 1e-4}, 1);
		mv::Iterate M(Obj, {Manifold.Share()});
		const bool converged = mv::AugmentedLagrangian(1, 3.3, 0.8, {1e-5}, 4, 1)(mv::TruncatedNewton)(
				M, tr, cg, Tolerance, 10, 1
		);
		__Check_Result__
	};

	void testLBFGS(){
		mv::Iterate M(Obj, {Manifold.Share()});
		const bool converged = mv::AugmentedLagrangian(1, 3.3, 0.8, {1e-5}, 4, 1)(mv::LBFGS)(
				M, Tolerance,
				10, 20, 0.1, 0.75, 7, 1
		);
		__Check_Result__
	};

	void testLanczos(){
		Obj.Lambda = { Lambda };
		mv::Iterate M(Obj, {Manifold.Share()});
		M.setPoint({Solution}, 1);
		M.Func->Calculate(M.getPoint(), {0, 1, 2});
		M.setGradient();
		const auto [Evals, Evecs] = mv::Lanczos(M, M.getDimension() - 1, 1);
		__Check_Stability__
	};
};

int main(){
	TestRayleighLagrange().testTruncatedNewton();
	TestRayleighLagrange().testLBFGS();
	TestRayleighLagrange().testLanczos();
}
