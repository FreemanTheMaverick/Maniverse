#include <Eigen/Dense>
#include <cmath>
#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <Maniverse/Manifold/Euclidean.h>
#include <Maniverse/Optimizer/TruncatedNewton.h>
#include <Maniverse/Optimizer/LBFGS.h>
#include <Maniverse/Optimizer/Anderson.h>

// Quadratic minimization
// Finding the bottom of a quadratic form
// Minimize L(x) = x.t A x
// A \in SPD(10), nearly diagonal
// x \in R(10)

namespace mv = Maniverse;

class UnpreconObjQuadratic: public mv::Objective{ public:
	Eigen::MatrixXd A = Eigen::MatrixXd::Zero(10, 10);

	UnpreconObjQuadratic(){
		const double data[] = {
			#include "Sym10.txt"
		};
		std::memcpy(A.data(), &data, 10 * 10 * 8);
		A = A * A + Eigen::MatrixXd::Identity(10, 10) * 0.01; // Constructing a SPD matrix whose diagonal elements dominate
		for ( int i = 0; i < 10; i++ ) for ( int j = 0; j < 10; j++ )
			if ( i != j ) A(i, j) *= 0.01;
	};

	virtual void Calculate(std::vector<Eigen::MatrixXd> x, int /*derivative*/) override{
		Value = x[0].cwiseProduct( A * x[0] ).sum();
		Gradient = { 2 * A * x[0] };
	};

	std::vector<std::vector<Eigen::MatrixXd>> Hessian(std::vector<Eigen::MatrixXd> v) const override{
		return std::vector<std::vector<Eigen::MatrixXd>>{{ 2 * A * v[0] }};
	};
};

class PreconObjQuadratic: public UnpreconObjQuadratic{ public:
	Eigen::MatrixXd Ainv = ( 2 * A ).diagonal().cwiseAbs().cwiseInverse().asDiagonal();
	Eigen::MatrixXd Asqrt = ( 2 * A ).diagonal().cwiseAbs().cwiseSqrt().asDiagonal();
	Eigen::MatrixXd Ainvsqrt = ( 2 * A ).diagonal().cwiseAbs().cwiseInverse().cwiseSqrt().asDiagonal();

	std::vector<std::vector<Eigen::MatrixXd>> Preconditioner(std::vector<Eigen::MatrixXd> v) const override{
		return std::vector<std::vector<Eigen::MatrixXd>>{{ Ainv * v[0] }};
	};

	std::vector<std::vector<Eigen::MatrixXd>> PreconditionerSqrt(std::vector<Eigen::MatrixXd> v) const override{
		return std::vector<std::vector<Eigen::MatrixXd>>{{ Ainvsqrt * v[0] }};
	};

	std::vector<std::vector<Eigen::MatrixXd>> PreconditionerInvSqrt(std::vector<Eigen::MatrixXd> v) const override{
		return std::vector<std::vector<Eigen::MatrixXd>>{{ Asqrt * v[0] }};
	};
};

class AndersonObjQuadratic: public UnpreconObjQuadratic{ public:
	void Calculate(std::vector<Eigen::MatrixXd> x, int /*derivative*/) override{
		UnpreconObjQuadratic::Calculate(x, 0);
		Gradient = { - 2 * A * x[0] };
	};
};

#define __Check_Result__\
	std::cout << typeid(*this).name() << " " << __func__ << " ";\
	if ( converged ){\
		if ( ( M.Ms[0]->P ).cwiseAbs().maxCoeff() < 1e-5 ){\
			std::cout << "\033[32mSuccess!\033[0m" << std::endl;\
		}else std::cout << "\033[31mFailed: Incorrect solution!\033[0m" << std::endl;\
	}else std::cout << "\033[31mFailed: Not converged!\033[0m" << std::endl;

class TestQuadratic{ public:
	UnpreconObjQuadratic UnpreconObj = UnpreconObjQuadratic();
	PreconObjQuadratic PreconObj = PreconObjQuadratic();
	AndersonObjQuadratic AndersonObj = AndersonObjQuadratic();
	mv::Euclidean Manifold = mv::Euclidean(Eigen::MatrixXd::Zero(10, 1));
	std::tuple<double, double, double> Tolerance = {1.e-5, 1.e-5, 1.e-5};
	mv::TrustRegion TrustRegion = mv::TrustRegion();

	TestQuadratic(){
		Eigen::MatrixXd from0to9(10, 1);
		from0to9 << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9;
		Manifold = mv::Euclidean(from0to9);
	};

	void testUnpreconTruncatedNewton(){
		mv::Iterate M(UnpreconObj, {Manifold.Share()}, true);
		const bool converged = mv::TruncatedNewton(
				M, TrustRegion, Tolerance,
				0.001, 21, 1
		);
		__Check_Result__
	};

	void testPreconTruncatedNewton(){
		mv::Iterate M(PreconObj, {Manifold.Share()}, true);
		const bool converged = mv::TruncatedNewton(
				M, TrustRegion, Tolerance,
				0.001, 19, 1
		);
		__Check_Result__
	};

	void testUnpreconLBFGS(){
		mv::Iterate M(UnpreconObj, {Manifold.Share()}, true);
		const bool converged = mv::LBFGS(
				M, Tolerance,
				20, 11, 0.1, 0.75, 5, 1
		);
		__Check_Result__
	};

	void testPreconLBFGS(){
		mv::Iterate M(PreconObj, {Manifold.Share()}, true);
		const bool converged = mv::LBFGS(
				M, Tolerance,
				20, 7, 0.1, 0.75, 5, 1
		);
		__Check_Result__
	};

	void testAnderson(){
		mv::Iterate M(AndersonObj, {Manifold.Share()}, true);
		const bool converged = mv::Anderson(
				M, Tolerance,
				0.2, 6, 12, 1
		);
		__Check_Result__
	};
};

int main(){
	TestQuadratic().testUnpreconTruncatedNewton();
	TestQuadratic().testPreconTruncatedNewton();
	TestQuadratic().testUnpreconLBFGS();
	TestQuadratic().testPreconLBFGS();
	TestQuadratic().testAnderson();
}
