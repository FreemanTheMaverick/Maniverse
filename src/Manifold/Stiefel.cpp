#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#endif
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <cmath>
#include <functional>
#include <tuple>
#include <memory>

#include "../Macro.h"

#include "Stiefel.h"

Stiefel::Stiefel(EigenMatrix p, std::string geodesic): Manifold(p, geodesic){
	__Check_Geodesic__("EXACT", "POLAR")
	this->Name = "Stiefel("
		+ std::to_string(p.rows())
		+ ", "
		+ std::to_string(p.cols())
		+ ")";
	if ( ! ( p.transpose() * p ).isApprox(EigenOne(p.cols(), p.cols())) )
		throw std::runtime_error("A Stiefel matrix must fulfill Ut.U = I!");
}

int Stiefel::getDimension() const{
	const int n = this->P.rows();
	const int k = this->P.cols();
	return n * k - k * ( k + 1 ) / 2;
}

double Stiefel::Inner(EigenMatrix X, EigenMatrix Y) const{
	return X.cwiseProduct(Y).sum();
}

EigenMatrix Stiefel::Retract(EigenMatrix X) const{
	if ( this->Geodesic == "EXACT" ){
		EigenMatrix A = EigenZero(X.rows(), 2 * X.cols());
		A << this->P, X;
		EigenMatrix B = EigenZero(2 * X.cols(), 2 * X.cols());
		B.topLeftCorner(X.cols(), X.cols()) = B.bottomRightCorner(X.cols(), X.cols()) = this->P.transpose() * X;
		B.topRightCorner(X.cols(), X.cols()) = - X.transpose() * X;
		B.bottomLeftCorner(X.cols(), X.cols()) = EigenOne(X.cols(), X.cols());
		EigenMatrix C = EigenZero(2 * X.cols(), X.cols());
		C.topRows(X.cols()) = ( - this->P.transpose() * X ).exp();
		return A * B.exp() * C;
	}else if ( this->Geodesic == "POLAR" ){
		Eigen::BDCSVD<EigenMatrix> svd;
		svd.compute(this->P + X, Eigen::ComputeThinU | Eigen::ComputeFullV);
		return svd.matrixU() * svd.matrixV().transpose();
	}
	__Check_Geodesic_Func__
	return X;
}

inline static EigenMatrix Sylvester(EigenMatrix A, EigenMatrix Q){
	// https://discourse.mc-stan.org/t/solve-a-lyapunov-sylvester-equation-include-custom-c-function-using-eigen-library-possible/12688

	const EigenMatrix B = A.transpose();

	Eigen::ComplexSchur<EigenMatrix> SchurA(A);
	const Eigen::MatrixXcd R = SchurA.matrixT();
	const Eigen::MatrixXcd U = SchurA.matrixU();

	Eigen::ComplexSchur<EigenMatrix> SchurB(B);
	const Eigen::MatrixXcd S = SchurB.matrixT();
	const Eigen::MatrixXcd V = SchurB.matrixU();

	const Eigen::MatrixXcd F = U.adjoint() * Q * V;
	const Eigen::MatrixXcd Y = Eigen::internal::matrix_function_solve_triangular_sylvester(R, S, F);
	const Eigen::MatrixXcd X = U * Y * V.adjoint();

	return X.real();
}

EigenMatrix Stiefel::InverseRetract(Manifold& N) const{
	// https://doi.org/10.1109/TSP.2012.2226167
	__Check_Log_Map__
	const EigenMatrix p = this->P;
	const EigenMatrix q = N.P;
	if ( this->Geodesic == "POLAR" ){ // Algorithm 2
		const EigenMatrix M = p.transpose() * q;
		const EigenMatrix S = Sylvester(M, 2 * EigenOne(p.cols(), p.cols()));
		return q * S - p;
	}
	__Check_Geodesic_Func__
	return q;
}

EigenMatrix Stiefel::TransportTangent(EigenMatrix Y, EigenMatrix Z) const{
	// Transport Y along Z
	// Section 3.5, https://doi.org/10.1007/s10589-016-9883-4
	if ( this->Geodesic == "POLAR" ){
		const EigenMatrix IplusZtZ = EigenOne(Z.cols(), Z.cols()) + Z.transpose() * Z;
		Eigen::SelfAdjointEigenSolver<EigenMatrix> es(IplusZtZ);
		const EigenMatrix A = es.operatorSqrt();
		const EigenMatrix Ainv = es.operatorInverseSqrt();
		const EigenMatrix RZ = this->Retract(Z);
		const EigenMatrix RZtY = RZ.transpose() * Y;
		const EigenMatrix Q = RZtY - RZtY.transpose();
		const EigenMatrix Lambda = Sylvester(A, Q);
		return RZ * Lambda + ( EigenOne(Z.rows(), Z.rows()) - RZ * RZ.transpose() ) * Y * Ainv;
	}
	__Check_Geodesic_Func__
	return Y;
}

EigenMatrix Stiefel::TransportManifold(EigenMatrix X, Manifold& N) const{
	__Check_Vec_Transport__
	const EigenMatrix Z = this->InverseRetract(N);
	return this->TransportTangent(X, Z);
}

inline static EigenMatrix TangentProjection(EigenMatrix P, EigenMatrix A){
	//https://juliamanifolds.github.io/Manifolds.jl/stable/manifolds/stiefel
	const EigenMatrix PtA = P.transpose() * A;
	const EigenMatrix SymPtA = 0.5 * ( PtA + PtA.transpose() );
	return A - P * SymPtA;
}

EigenMatrix Stiefel::TangentProjection(EigenMatrix X) const{
	return ::TangentProjection(this->P, X);
}

EigenMatrix Stiefel::TangentPurification(EigenMatrix X) const{
	return ::TangentProjection(this->P, X);
}

void Stiefel::setPoint(EigenMatrix p, bool purify){
	if (purify){
		Eigen::BDCSVD<EigenMatrix> svd(p, Eigen::ComputeThinU | Eigen::ComputeThinV);
		p = svd.matrixU() * svd.matrixV().transpose();
	}
	this->P = p;
}

void Stiefel::getGradient(){
	this->Gr = this->TangentProjection(this->Ge);
}

std::function<EigenMatrix (EigenMatrix)> Stiefel::getHessian(std::function<EigenMatrix (EigenMatrix)> He, bool weingarten) const{
	//https://juliamanifolds.github.io/Manifolds.jl/stable/manifolds/stiefel
	const EigenMatrix P = this->P;
	const EigenMatrix tmp = this->Ge.transpose() * this->P + this->P.transpose() * this->Ge;
	if ( weingarten ) return [P, tmp, He](EigenMatrix v){
		return ::TangentProjection(P, He(v) - 0.5 * v * tmp);
	};
	else return [P, He](EigenMatrix v){
		return ::TangentProjection(P, He(v));
	};
}

std::unique_ptr<Manifold> Stiefel::Clone() const{
	return std::make_unique<Stiefel>(*this);
}

#ifdef __PYTHON__
void Init_Stiefel(pybind11::module_& m){
	pybind11::classh<Stiefel, Manifold>(m, "Stiefel")
		.def(pybind11::init<EigenMatrix, std::string>(), pybind11::arg("p"), pybind11::arg("geodesic") = "POLAR");
}
#endif
