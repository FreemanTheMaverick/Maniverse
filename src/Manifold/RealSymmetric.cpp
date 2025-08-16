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

inline static EigenMatrix Symmetrize(EigenMatrix X){
	return ( X + X.transpose() ) / 2;
}

RealSymmetric::RealSymmetric(EigenMatrix p, std::string geodesic): Euclidean(p, geodesic){
	this->Name = "RealSymmetric(" + std::to_string(p.rows()) + ", " + std::to_string(p.cols()) + ")";
}

int RealSymmetric::getDimension() const{
	return ( 1 + this->P.rows() ) * this->P.rows() / 2;
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
	pybind11::classh<RealSymmetric, Euclidean>(m, "RealSymmetric")
		.def(pybind11::init<EigenMatrix, std::string>(), pybind11::arg("p"), pybind11::arg("geodesic") = "EXACT");
}
#endif
