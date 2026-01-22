#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <cmath>
#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <Maniverse/Manifold/Stiefel.h>
#include <Maniverse/Optimizer/TruncatedNewton.h>
#include <Maniverse/Optimizer/LBFGS.h>
#include <Maniverse/Optimizer/Anderson.h>

// Orthogonal projection
// Finding the Stiefel matrix closest to the given matrix A
// Minimize L(C) = || C - A ||^2
// A \in R(10, 6)
// C \in St(10, 6)

namespace mv = Maniverse;

class ObjProjection: public mv::Objective{ public:
	Eigen::MatrixXd A = Eigen::MatrixXd::Zero(10, 6);

	ObjProjection(){
		const double data[] = {
			#include "Sym10.txt"
		};
		std::memcpy(A.data(), &data, 10 * 6 * 8);
	};

	virtual void Calculate(std::vector<Eigen::MatrixXd> C, int /*derivative*/) override{
		Value = ( C[0] - A ).squaredNorm();
		Gradient = { 2 * ( C[0] - A ) };
	};

	std::vector<Eigen::MatrixXd> Hessian(std::vector<Eigen::MatrixXd> V) const override{
		return std::vector<Eigen::MatrixXd>{ 2 * V[0] };
	};
};

class AndersonObjProjection: public ObjProjection{ public:
	void Calculate(std::vector<Eigen::MatrixXd> C, int /*derivative*/) override{
		ObjProjection::Calculate(C, 0);
		Gradient = { - 2 * ( C[0] - A ) };
	};
};

#define __Check_Result__\
	std::cout << typeid(*this).name() << " " << __func__ << " ";\
	if ( converged ){\
		if ( ( M.Ms[0]->P - Solution ).cwiseAbs().maxCoeff() < 1e-5 ){\
			std::cout << "\033[32mSuccess!\033[0m" << std::endl;\
		}else std::cout << "\033[31mFailed: Incorrect solution!\033[0m" << std::endl;\
	}else std::cout << "\033[31mFailed: Not converged!\033[0m" << std::endl;

class TestProjection{ public:
	ObjProjection Obj = ObjProjection();
	AndersonObjProjection AndersonObj = AndersonObjProjection();
	mv::Stiefel Manifold = mv::Stiefel(Eigen::MatrixXd::Identity(10, 6));
	std::tuple<double, double, double> Tolerance = {1.e-5, 1.e-5, 1.e-5};
	mv::TrustRegion TrustRegion = mv::TrustRegion();
	Eigen::MatrixXd Solution = Eigen::MatrixXd::Identity(10, 6);

	TestProjection(){
		Eigen::BDCSVD<Eigen::MatrixXd, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(Obj.A);
		const Eigen::MatrixXd U = svd.matrixU();
		const Eigen::MatrixXd V = svd.matrixV();
		Manifold = mv::Stiefel( U * V.transpose() * ( Obj.A.bottomRows(6) - Obj.A.bottomRows(6).transpose() ).exp() );
		Solution = U * V.transpose();
	};

	void testTruncatedNewton(){
		mv::Iterate M(Obj, {Manifold.Share()}, true);
		const bool converged = mv::TruncatedNewton(
				M, TrustRegion, Tolerance,
				0.001, 9, 1
		);
		__Check_Result__
	};

	void testLBFGS(){
		mv::Iterate M(Obj, {Manifold.Share()}, true);
		const bool converged = mv::LBFGS(
				M, Tolerance,
				20, 19, 0.1, 0.75, 5, 1
		);
		__Check_Result__
	};

	void testAnderson(){
		mv::Iterate M(AndersonObj, {Manifold.Share()}, true);
		const bool converged = mv::Anderson(
				M, Tolerance,
				0.2, 6, 28, 1
		);
		__Check_Result__
	};
};

int main(){
	TestProjection().testTruncatedNewton();
	TestProjection().testLBFGS();
	TestProjection().testAnderson();
}
