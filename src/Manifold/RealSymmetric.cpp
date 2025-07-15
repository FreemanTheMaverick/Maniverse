#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#endif
#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <cassert>
#include <memory>

#include "../Macro.h"

#include "RealSymmetric.h"

inline EigenMatrix Symmetrize(EigenMatrix X){
	return ( X + X.transpose() ) / 2;
}

RealSymmetric::RealSymmetric(EigenMatrix p, bool matrix_free): Manifold(p, matrix_free){
	this->Name = "RealSymmetric";
}

int RealSymmetric::getDimension() const{
	return this->P.size();
}

double RealSymmetric::Inner(EigenMatrix X, EigenMatrix Y) const{
	return (X.cwiseProduct(Y)).sum();
}

EigenMatrix RealSymmetric::Exponential(EigenMatrix X) const{
	return this->P + X;
}

EigenMatrix RealSymmetric::Logarithm(Manifold& N) const{
	return N.P - this->P;
}

EigenMatrix RealSymmetric::TangentProjection(EigenMatrix A) const{
	return Symmetrize(A);
}

EigenMatrix RealSymmetric::TangentPurification(EigenMatrix A) const{
	return Symmetrize(A);
}

void RealSymmetric::Update(EigenMatrix p, [[maybe_unused]] bool purify){
	this->P = Symmetrize(p);
}

void RealSymmetric::getGradient(){
	this->Gr = Symmetrize(this->Ge);
}

void RealSymmetric::getHessian(){
	const std::function<EigenMatrix (EigenMatrix)> He = this->He;
	this->Hr = [He](EigenMatrix v){
		return Symmetrize(He(v));
	};
}

std::unique_ptr<Manifold> RealSymmetric::Clone() const{
	return std::make_unique<RealSymmetric>(*this);
}

#ifdef __PYTHON__
void Init_RealSymmetric(pybind11::module_& m){
	pybind11::class_<RealSymmetric, Manifold>(m, "RealSymmetric")
		.def(pybind11::init<EigenMatrix, bool>());
}
#endif
