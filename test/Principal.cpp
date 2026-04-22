#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <Maniverse/Manifold/Flag.h>
#include <Maniverse/LinearSolver/ConjugateGradient.h>
#include <Maniverse/Optimizer/TruncatedNewton.h>
#include <Maniverse/Optimizer/LBFGS.h>
#include <Maniverse/Diagonalizer/Lanczos.h>

// Principal component analysis
// Finding the space spanned by the highest 5 eigenvectors
// Maximize L(C) = Tr[ C.t A C ]
// A \in Sym(10)
// C \in Flag(5; 10) = Gr(5; 10)

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

#define __Check_Stability__\
	std::cout << typeid(*this).name() << " " << __func__ << " ";\
	for ( int i = 0; i < (int)Evecs.size(); i++ ){\
		const double residual = ( M.ConstraintProjectedHessian(Evecs[i]) - Evals[i] * Evecs[i] ).norm();\
		if ( residual > 1e-5 ) goto IncorrectCurvature;\
	}\
	std::cout << "\033[32mSuccess!\033[0m" << std::endl; return;\
	IncorrectCurvature: std::cout << "\033[31mFailed: Eigenvalue equation is violated!\033[0m" << std::endl;

class TestPrincipal{ public:
	ObjPrincipal Obj = ObjPrincipal();
	mv::Flag Manifold = mv::Flag(Eigen::MatrixXd::Identity(10, 5));
	std::tuple<double, double, double> Tolerance = {1.e-5, 1.e-5, 1.e-5};
	Eigen::MatrixXd Solution = Eigen::MatrixXd::Identity(10, 5);

	TestPrincipal(){
		Manifold.setBlockParameters({5});
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
		es.compute(Obj.A);
		Solution = es.eigenvectors().rightCols(5);
	};

	void testTruncatedNewton(){
		mv::Iterate M(Obj, {Manifold.Share()});
		mv::TrustRegion tr;
		mv::ConjugateGradient cg(3, 1, {1e-4, 1e-4}, 1);
		const bool converged = mv::TruncatedNewton(
				M, tr, cg, Tolerance, 8, 1
		);
		__Check_Result__
	};

	void testLBFGS(){
		mv::Iterate M(Obj, {Manifold.Share()});
		const bool converged = mv::LBFGS(
				M, Tolerance,
				10, 43, 0.1, 0.75, 5, 1
		);
		__Check_Result__
	};

	void testLanczos(){
		mv::Iterate M(Obj, {Manifold.Share()});
		M.setPoint({Solution}, 1);
		M.Func->Calculate(M.getPoint(), {0, 1, 2});
		M.setGradient();
		const auto [Evals, Evecs] = mv::Lanczos(M, M.getDimension(), 1);
		__Check_Stability__
	};
};

int main(){
	TestPrincipal().testTruncatedNewton();
	TestPrincipal().testLBFGS();
	TestPrincipal().testLanczos();
}
