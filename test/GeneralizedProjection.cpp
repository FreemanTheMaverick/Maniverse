#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <vector>
#include <iostream>
#include <Maniverse/Manifold/GeneralizedStiefel.h>
#include <Maniverse/LinearSolver/ConjugateGradient.h>
#include <Maniverse/LinearSolver/MinRes.h>
#include <Maniverse/Optimizer/Newton.h>
#include <Maniverse/Optimizer/LBFGS.h>
#include <Maniverse/Diagonalizer/Lanczos.h>

// Orthogonal projection
// Finding the generalized Stiefel matrix closest to the given matrix A
// Minimize L(C) = || C - A ||^2_B
// A \in R(10, 6)
// B \in SPD(10)
// C \in St_B(10, 6)

namespace mv = Maniverse;

class ObjProjection: public mv::Function{ public:
	Eigen::MatrixXd A = Eigen::MatrixXd::Zero(10, 6);
	Eigen::MatrixXd B = Eigen::MatrixXd::Zero(10, 10);

	ObjProjection(){
		const double data[] = {
			#include "Sym10.txt"
		};
		std::memcpy(A.data(), &data, 10 * 6 * 8);
		std::memcpy(B.data(), &data, 10 * 10 * 8);
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(B * B);
		B = es.operatorSqrt();
	};

	virtual void Calculate(std::vector<Eigen::MatrixXd> C, std::vector<int> derivatives) override{
		const Eigen::MatrixXd C_minus_A = C[0] - A;
		if ( std::count(derivatives.begin(), derivatives.end(), 0) ){
			Value = C_minus_A.cwiseProduct(B * C_minus_A).sum();
		}
		if ( std::count(derivatives.begin(), derivatives.end(), 1) ){
			Gradient = { 2 * B * C_minus_A };
		}
	};

	std::vector<Eigen::MatrixXd> Hessian(std::vector<Eigen::MatrixXd> V) const override{
		return std::vector<Eigen::MatrixXd>{ 2 * B * V[0] };
	};
};

#define __Check_Result__\
	std::cout << typeid(*this).name() << " " << __func__ << " ";\
	if ( converged ){\
		if ( ( M.Manifolds[0]->P - Solution ).cwiseAbs().maxCoeff() < 1e-5 ){\
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

class TestProjection{ public:
	ObjProjection Obj = ObjProjection();
	mv::GeneralizedStiefel Manifold = mv::GeneralizedStiefel(
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(Obj.B).operatorInverseSqrt().leftCols(6),
		Obj.B
	);
	std::array<double, 3> Tolerance = {1.e-5, 1.e-5, 1.e-5};
	Eigen::MatrixXd Solution = Eigen::MatrixXd::Identity(10, 6);

	TestProjection(){
		Eigen::LLT<Eigen::MatrixXd> llt(Obj.B);
		Eigen::MatrixXd L = llt.matrixL();
		Eigen::BDCSVD<Eigen::MatrixXd, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(L.transpose() * Obj.A);
		const Eigen::MatrixXd U = svd.matrixU();
		const Eigen::MatrixXd V = svd.matrixV();
		Solution = L.transpose().inverse() * U * V.transpose();
		Manifold = mv::GeneralizedStiefel( Solution * ( Obj.A.bottomRows(6) - Obj.A.bottomRows(6).transpose() ).exp(), Obj.B );
	};

	void testNewtonCG(){
		mv::Iterate M(Obj, {Manifold.Share()});
		mv::TrustRegion tr;
		mv::ConjugateGradient cg(1, {1e-4, 1e-4}, M.getDimension(), 1);
		mv::initLinearSolverForNewton(cg, M);
		const bool converged = mv::Newton(
				M, tr, cg, Tolerance, 9, 1
		);
		__Check_Result__
	};

	void testNewtonMR(){
		mv::Iterate M(Obj, {Manifold.Share()});
		mv::TrustRegion tr;
		mv::MinRes mr(1, {1e-4, 1e-4}, M.getDimension(), 1);
		mv::initLinearSolverForNewton(mr, M);
		const bool converged = mv::Newton(
				M, tr, mr, Tolerance, 10, 1
		);
		__Check_Result__
	};

	void testLBFGS(){
		mv::Iterate M(Obj, {Manifold.Share()});
		const bool converged = mv::LBFGS(
				M, Tolerance,
				20, 35, 0.1, 0.75, 5, 1
		);
		__Check_Result__
	};

	void testLanczos(){
		mv::Iterate M(Obj, {Manifold.Share()});
		M.setPoint({Solution}, 1);
		M.Calculate(M.getPoint(), {0, 1, 2});
		M.setGradient();
		const auto [Evals, Evecs] = mv::Lanczos(M, M.getDimension(), 0, 0, 1);
		__Check_Stability__
	};
};

int main(){
	TestProjection().testNewtonCG();
	TestProjection().testNewtonMR();
	TestProjection().testLBFGS();
	TestProjection().testLanczos();
}
