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

Stiefel::Stiefel(EigenMatrix p): Manifold(p){
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

EigenMatrix Stiefel::Exponential(EigenMatrix X) const{
	EigenMatrix A = EigenZero(X.rows(), 2 * X.cols());
	A << this->P, X;
	EigenMatrix B = EigenZero(2 * X.cols(), 2 * X.cols());
	B.topLeftCorner(X.cols(), X.cols()) = B.bottomRightCorner(X.cols(), X.cols()) = this->P.transpose() * X;
	B.topRightCorner(X.cols(), X.cols()) = - X.transpose() * X;
	B.bottomLeftCorner(X.cols(), X.cols()) = EigenOne(X.cols(), X.cols());
	EigenMatrix C = EigenZero(2 * X.cols(), X.cols());
	C.topRows(X.cols()) = ( - this->P.transpose() * X ).exp();
	return A * B.exp() * C;
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
		.def(pybind11::init<EigenMatrix>());
}
#endif
