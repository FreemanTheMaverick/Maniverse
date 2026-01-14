#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#endif
#include <Eigen/Dense>
#include <cmath>
#include <memory>

#include "../Macro.h"

#include "RealSymmetric.h"

namespace Maniverse{

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

EigenMatrix RealSymmetric::getHessian(EigenMatrix HeX, EigenMatrix /*X*/, bool /*weingarten*/) const{
	return Symmetrize(HeX);
}

std::shared_ptr<Manifold> RealSymmetric::Share() const{
	return std::make_shared<RealSymmetric>(*this);
}

#ifdef __PYTHON__
void Init_RealSymmetric(pybind11::module_& m){
	pybind11::classh<RealSymmetric, Euclidean>(m, "RealSymmetric")
		.def(pybind11::init<EigenMatrix, std::string>(), pybind11::arg("p"), pybind11::arg("geodesic") = "EXACT");
}
#endif

}
