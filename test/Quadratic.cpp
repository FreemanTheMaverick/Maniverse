#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <Maniverse/Manifold/Euclidean.h>
#include <Maniverse/LinearSolver/ConjugateGradient.h>
#include <Maniverse/LinearSolver/MinRes.h>
#include <Maniverse/Optimizer/Newton.h>
#include <Maniverse/Optimizer/LBFGS.h>
#include <Maniverse/Optimizer/Anderson.h>
#include <Maniverse/Diagonalizer/Lanczos.h>

// Quadratic minimization
// Finding the bottom of a quadratic form
// Minimize L(x) = x.t A x
// A \in SPD(10), nearly diagonal
// x \in R(10)

namespace mv = Maniverse;

class ObjQuadratic: public mv::Function{ public:
	Eigen::MatrixXd A = Eigen::MatrixXd::Zero(10, 10);
	Eigen::MatrixXd Ax = Eigen::MatrixXd::Zero(10, 1); // Temporary variable to reuse
	Eigen::VectorXd Ainv = Eigen::VectorXd::Zero(10);
	Eigen::VectorXd Asqrt = Eigen::VectorXd::Zero(10);
	Eigen::VectorXd Ainvsqrt = Eigen::VectorXd::Zero(10);

	ObjQuadratic(){
		const double data[] = {
			#include "Sym10.txt"
		};
		std::memcpy(A.data(), &data, 10 * 10 * 8);
		A = A * A + Eigen::MatrixXd::Identity(10, 10) * 0.01; // Constructing a SPD matrix whose diagonal elements dominate
		Ainv = ( 2 * A ).diagonal().cwiseAbs().cwiseInverse();
		Asqrt = ( 2 * A ).diagonal().cwiseAbs().cwiseSqrt();
		Ainvsqrt = ( 2 * A ).diagonal().cwiseAbs().cwiseInverse().cwiseSqrt();
		for ( int i = 0; i < 10; i++ ) for ( int j = 0; j < 10; j++ )
			if ( i != j ) A(i, j) *= 0.01;
	};

	virtual void Calculate(std::vector<Eigen::MatrixXd> x, std::vector<int> derivatives) override{
		if ( std::count(derivatives.begin(), derivatives.end(), 0) ){
			Ax = A * x[0];
			Value = x[0].cwiseProduct(Ax).sum();
		}
		if ( std::count(derivatives.begin(), derivatives.end(), 1) ){
			Gradient = { 2 * Ax };
		}
	};

	std::vector<Eigen::MatrixXd> Hessian(std::vector<Eigen::MatrixXd> v) const override{
		return std::vector<Eigen::MatrixXd>{ 2 * A * v[0] };
	};
};

class AndersonObjQuadratic: public ObjQuadratic{ public:
	void Calculate(std::vector<Eigen::MatrixXd> x, std::vector<int> derivatives) override{
		ObjQuadratic::Calculate(x, derivatives);
		if ( std::count(derivatives.begin(), derivatives.end(), 1) ){
			Gradient = { - 2 * A * x[0] };
		}
	};
};

#define __Check_Result__\
	std::cout << typeid(*this).name() << " " << __func__ << " ";\
	if ( converged ){\
		if ( ( M.Manifolds[0]->P ).cwiseAbs().maxCoeff() < 1e-5 ){\
			std::cout << "\033[32mSuccess!\033[0m" << std::endl;\
		}else std::cout << "\033[31mFailed: Incorrect solution!\033[0m" << std::endl;\
	}else std::cout << "\033[31mFailed: Not converged!\033[0m" << std::endl;

#define __Check_Stability__\
	std::cout << typeid(*this).name() << " " << __func__ << " ";\
	for ( int i = 0; i < (int)Evecs.size(); i++ ){\
		const double residual = ( M.Hessian(Evecs[i]) - Evals[i] * Evecs[i] ).norm();\
		if ( residual > 1e-5 ) goto IncorrectCurvature;\
	}\
	std::cout << "\033[32mSuccess!\033[0m" << std::endl; return;\
	IncorrectCurvature: std::cout << "\033[31mFailed: Eigenvalue equation is violated!\033[0m" << std::endl;

class TestQuadratic{ public:
	ObjQuadratic Obj = ObjQuadratic();
	AndersonObjQuadratic AndersonObj = AndersonObjQuadratic();
	mv::Euclidean Manifold = mv::Euclidean(Eigen::MatrixXd::Zero(10, 1));
	std::array<double, 3> Tolerance = {1.e-5, 1.e-5, 1.e-5};

	TestQuadratic(){
		Eigen::MatrixXd from0to9(10, 1);
		from0to9 << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9;
		Manifold = mv::Euclidean(from0to9);
	};

	void testUnpreconNewtonCG(){
		mv::Iterate M(Obj, {Manifold.Share()});
		mv::TrustRegion tr;
		mv::ConjugateGradient cg(1, {1e-4, 1e-4}, M.getDimension(), 1);
		mv::initLinearSolverForNewton(cg, M);
		M.Preconditioner = [&Ainv = Obj.Ainv](Eigen::VectorXd v) -> Eigen::VectorXd {
			return Ainv.cwiseProduct(v).eval();
		};
		const bool converged = mv::Newton(
				M, tr, cg, Tolerance, 20, 1
		);
		__Check_Result__
	};

	void testUnpreconNewtonMR(){
		mv::Iterate M(Obj, {Manifold.Share()});
		mv::TrustRegion tr;
		mv::MinRes mr(1, {1e-8, 1e-8}, M.getDimension(), 1);
		mv::initLinearSolverForNewton(mr, M);
		const bool converged = mv::Newton(
				M, tr, mr, Tolerance, 20, 1
		);
		__Check_Result__
	};


	void testPreconNewtonCG(){
		mv::Iterate M(Obj, {Manifold.Share()});
		mv::TrustRegion tr;
		mv::ConjugateGradient cg(1, {1e-4, 1e-4}, M.getDimension(), 1);
		mv::initLinearSolverForNewton(cg, M);
		M.Preconditioner = [&Ainv = Obj.Ainv](Eigen::VectorXd v) -> Eigen::VectorXd {
			return Ainv.cwiseProduct(v).eval();
		};
		const bool converged = mv::Newton(
				M, tr, cg, Tolerance, 19, 1
		);
		__Check_Result__
	};

	void testPreconNewtonMR(){
		mv::Iterate M(Obj, {Manifold.Share()});
		mv::TrustRegion tr;
		mv::MinRes mr(1, {1e-8, 1e-8}, M.getDimension(), 1);
		mv::initLinearSolverForNewton(mr, M);
		M.Preconditioner = [&Ainv = Obj.Ainv](Eigen::VectorXd v) -> Eigen::VectorXd {
			return Ainv.cwiseProduct(v).eval();
		};
		const bool converged = mv::Newton(
				M, tr, mr, Tolerance, 20, 1
		);
		__Check_Result__
	};

	void testUnpreconLBFGS(){
		mv::Iterate M(Obj, {Manifold.Share()});
		const bool converged = mv::LBFGS(
				M, Tolerance,
				20, 11, 0.1, 0.75, 5, 1
		);
		__Check_Result__
	};

	void testPreconLBFGS(){
		mv::Iterate M(Obj, {Manifold.Share()});
		M.PreconditionerSqrt = [&Ainvsqrt = Obj.Ainvsqrt](Eigen::VectorXd v) -> Eigen::VectorXd {
			return Ainvsqrt.cwiseProduct(v).eval();
		};
		M.PreconditionerInvSqrt = [&Asqrt = Obj.Asqrt](Eigen::VectorXd v) -> Eigen::VectorXd {
			return Asqrt.cwiseProduct(v).eval();
		};
		const bool converged = mv::LBFGS(
				M, Tolerance,
				20, 7, 0.1, 0.75, 5, 1
		);
		__Check_Result__
	};

	void testAnderson(){
		mv::Iterate M(AndersonObj, {Manifold.Share()});
		const bool converged = mv::Anderson(
				M, Tolerance,
				0.2, 6, 12, 1
		);
		__Check_Result__
	};

	void testLanczos(){
		mv::Iterate M(Obj, {Manifold.Share()});
		M.setPoint({Eigen::MatrixXd::Zero(10, 1)}, 1);
		M.Calculate(M.getPoint(), {0, 1, 2});
		M.setGradient();
		const auto [Evals, Evecs] = mv::Lanczos(M, M.getDimension(), 0, 0, 1);
		__Check_Stability__
	};
};

int main(){
	TestQuadratic().testUnpreconNewtonCG();
	TestQuadratic().testUnpreconNewtonMR();
	TestQuadratic().testPreconNewtonCG();
	TestQuadratic().testPreconNewtonMR();
	TestQuadratic().testUnpreconLBFGS();
	TestQuadratic().testPreconLBFGS();
	TestQuadratic().testAnderson();
	TestQuadratic().testLanczos();
}
