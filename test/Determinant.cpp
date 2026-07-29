#include <Eigen/Dense>
#include <iostream>
#include <Maniverse/Manifold/Flag.h>
#include <Maniverse/LinearSolver/ConjugateGradient.h>
#include <Maniverse/Optimizer/Newton.h>
#include <Maniverse/Optimizer/LBFGS.h>
#include <Maniverse/Diagonalizer/Lanczos.h>

// Alignment of two spaces
// Finding the maximal overlap of two spaces
// Minimize L(C) = det[ C0.t C ] ( -1 )
// C0, C \in Flag(5; 10) = Gr(5; 10)

namespace mv = Maniverse;

#include "Determinant.h"

#define __Check_Result__\
	std::cout << typeid(*this).name() << " " << __func__ << " ";\
	if ( converged ){\
		if ( ( M.Ms[0]->P * M.Ms[0]->P.transpose() - Solution * Solution.transpose() ).cwiseAbs().maxCoeff() < 1e-5 ){\
			std::cout << "\033[32mSuccess!\033[0m" << std::endl;\
		}else std::cout << "\033[31mFailed: Incorrect solution!\033[0m" << std::endl;\
	}else std::cout << "\033[31mFailed: Not converged!\033[0m" << std::endl;

#define __Check_Stability__\
	std::cout << typeid(*this).name() << " " << __func__ << " ";\
	const double residual = ( M.Hessian(Evecs[0]) - Evals[0] * Evecs[0] ).norm();\
	if ( residual > 1e-5 ) std::cout << "\033[31mFailed: Eigenvalue equation is violated!\033[0m" << std::endl;\
	else std::cout << "\033[32mSuccess!\033[0m" << std::endl;\
	return;

class TestDeterminant{ public:
	ObjDeterminant Obj = ObjDeterminant(Eigen::MatrixXd::Zero(10, 5));
	mv::Flag Manifold = mv::Flag(Eigen::MatrixXd::Identity(10, 5));
	std::tuple<double, double, double> Tolerance = {1.e-5, 1.e-5, 1.e-5};
	Eigen::MatrixXd Solution = Eigen::MatrixXd::Zero(10, 5);

	TestDeterminant(){
		const double data[] = {
			#include "Sym10.txt"
		};
		const Eigen::MatrixXd A = Eigen::Map<const Eigen::MatrixXd>(data, 10, 10);
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);
		const Eigen::MatrixXd eigvecs = es.eigenvectors();
		Obj = ObjDeterminant(eigvecs.leftCols(5));
		Solution = - eigvecs.leftCols(5);
		Manifold.setBlockParameters({5});
	};

	void testNewtonCG(){
		mv::Iterate M(Obj, {Manifold.Share()});
		mv::TrustRegion tr;
		mv::ConjugateGradient cg(M, 0, 1, {1e-4, 1e-4}, M.getDimension(), 1);
		const bool converged = mv::Newton(
				M, tr, cg, Tolerance, 10, 1
		);
		__Check_Result__
	};

	void testLBFGS(){
		mv::Iterate M(Obj, {Manifold.Share()});
		const bool converged = mv::LBFGS(
				M, Tolerance,
				10, 10, 0.1, 0.75, 15, 1
		);
		__Check_Result__
	};

	void testLanczos(){
		mv::Iterate M(Obj, {Manifold.Share()});
		M.setPoint({Solution}, 1);
		M.Func->Calculate(M.getPoint(), {0, 1, 2});
		M.setGradient();
		const auto [Evals, Evecs] = mv::Lanczos(M, 1, 0, 0, 1);
		__Check_Stability__
	};
};

int main(){
	TestDeterminant().testNewtonCG();
	TestDeterminant().testLBFGS();
	TestDeterminant().testLanczos();
}
