#include <Eigen/Dense>
#include <cmath>
#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <Maniverse/Manifold/Flag.h>
#include <Maniverse/Optimizer/TruncatedNewton.h>
#include <Maniverse/Optimizer/LBFGS.h>

// Principal component analysis
// Finding the space spanned by the highest 5 eigenvectors
// Maximize L(C) = Tr[ C.t A C ]
// A \in Sym(10)
// C \in Flag(1, 2, 3, 4, 5; 10)

namespace mv = Maniverse;

class ObjPrincipal: public mv::Objective{ public:
	Eigen::MatrixXd A = Eigen::MatrixXd::Zero(10, 10);

	ObjPrincipal(){
		const double data[] = {
			#include "Sym10.txt"
		};
		std::memcpy(A.data(), &data, 10 * 10 * 8);
	};

	void Calculate(std::vector<Eigen::MatrixXd> C, std::vector<int> derivatives) override{
		if ( std::count(derivatives.begin(), derivatives.end(), 0) ){
			Value = - C[0].cwiseProduct( A * C[0] ).sum();
		}
		if ( std::count(derivatives.begin(), derivatives.end(), 1) ){
			Gradient = { - 2 * A * C[0] };
		}
	};

	std::vector<Eigen::MatrixXd> Hessian(std::vector<Eigen::MatrixXd> V) const override{
		return std::vector<Eigen::MatrixXd>{ - 2 * A * V[0] };
	};
};

#define __Check_Result__\
	std::cout << typeid(*this).name() << " " << __func__ << " ";\
	if ( converged ){\
		if ( ( M.Ms[0]->P * M.Ms[0]->P.transpose() - Solution * Solution.transpose() ).cwiseAbs().maxCoeff() < 1e-5 ){\
			std::cout << "\033[32mSuccess!\033[0m" << std::endl;\
		}else std::cout << "\033[31mFailed: Incorrect solution!\033[0m" << std::endl;\
	}else std::cout << "\033[31mFailed: Not converged!\033[0m" << std::endl;

class TestPrincipal{ public:
	ObjPrincipal Obj = ObjPrincipal();
	mv::Flag Manifold = mv::Flag(Eigen::MatrixXd::Identity(10, 5));
	std::tuple<double, double, double> Tolerance = {1.e-5, 1.e-5, 1.e-5};
	mv::TrustRegion TrustRegion = mv::TrustRegion();
	Eigen::MatrixXd Solution = Eigen::MatrixXd::Identity(10, 5);

	TestPrincipal(){
		Manifold.setBlockParameters({ 1, 1, 1, 1, 1 });
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
		es.compute(Obj.A);
		Solution = es.eigenvectors().rightCols(5);
	};

	void testTruncatedNewton(){
		mv::Iterate M(Obj, {Manifold.Share()}, true);
		const bool converged = mv::TruncatedNewton(
				M, TrustRegion, Tolerance,
				0.001, 13, 1
		);
		__Check_Result__
	};

	void testLBFGS(){
		mv::Iterate M(Obj, {Manifold.Share()}, true);
		const bool converged = mv::LBFGS(
				M, Tolerance,
				10, 46, 0.1, 0.75, 5, 1
		);
		__Check_Result__
	};
};

int main(){
	TestPrincipal().testTruncatedNewton();
	TestPrincipal().testLBFGS();
}
