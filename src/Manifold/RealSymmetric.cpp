#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#endif
#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <memory>

#include "../Macro.h"

#include "RealSymmetric.h"

inline EigenMatrix Symmetrize(EigenMatrix X){
	return ( X + X.transpose() ) / 2;
}

RealSymmetric::RealSymmetric(EigenMatrix p): Manifold(p){
	this->Name = "RealSymmetric(" + std::to_string(p.rows()) + ", " + std::to_string(p.cols()) + ")";
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

void RealSymmetric::setPoint(EigenMatrix p, bool /*purify*/){
	this->P = Symmetrize(p);
}

void RealSymmetric::getGradient(){
	this->Gr = Symmetrize(this->Ge);
}

std::function<EigenMatrix (EigenMatrix)> RealSymmetric::getHessian(std::function<EigenMatrix (EigenMatrix)> He, bool /*weingarten*/) const{
	return [He](EigenMatrix v){
		return Symmetrize(He(v));
	};
}

std::unique_ptr<Manifold> RealSymmetric::Clone() const{
	return std::make_unique<RealSymmetric>(*this);
}

#ifdef __PYTHON__
void Init_RealSymmetric(pybind11::module_& m){
	pybind11::classh<RealSymmetric, Manifold>(m, "RealSymmetric")
		.def(pybind11::init<EigenMatrix>());
}
#endif
