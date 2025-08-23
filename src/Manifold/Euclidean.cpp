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

#include "Euclidean.h"

namespace Maniverse{

Euclidean::Euclidean(EigenMatrix p, std::string geodesic): Manifold(p, geodesic){
	__Check_Geodesic__("EXACT")
	this->Name = "Euclidean(" + std::to_string(p.rows()) + ", " + std::to_string(p.cols()) + ")";
}

int Euclidean::getDimension() const{
	return this->P.size();
}

double Euclidean::Inner(EigenMatrix X, EigenMatrix Y) const{
	return (X.cwiseProduct(Y)).sum();
}

EigenMatrix Euclidean::Retract(EigenMatrix X) const{
	return this->P + X;
}

EigenMatrix Euclidean::InverseRetract(Manifold& N) const{
	__Check_Log_Map__
	return N.P - this->P;
}

EigenMatrix Euclidean::TransportTangent(EigenMatrix X, EigenMatrix /*Y*/) const{
	return X;
}

EigenMatrix Euclidean::TransportManifold(EigenMatrix X, Manifold& N) const{
	__Check_Log_Map__
	return X;
}

EigenMatrix Euclidean::TangentProjection(EigenMatrix A) const{
	return A;
}

EigenMatrix Euclidean::TangentPurification(EigenMatrix A) const{
	return A;
}

void Euclidean::setPoint(EigenMatrix p, bool /*purify*/){
	this->P = p;
}

void Euclidean::getGradient(){
	this->Gr = this->Ge;
}

std::function<EigenMatrix (EigenMatrix)> Euclidean::getHessian(std::function<EigenMatrix (EigenMatrix)> He, bool /*weingarten*/) const{
	return He;
}

std::unique_ptr<Manifold> Euclidean::Clone() const{
	return std::make_unique<Euclidean>(*this);
}

#ifdef __PYTHON__
void Init_Euclidean(pybind11::module_& m){
	pybind11::classh<Euclidean, Manifold>(m, "Euclidean")
		.def(pybind11::init<EigenMatrix, std::string>(), pybind11::arg("p"), pybind11::arg("geodesic") = "EXACT");
}
#endif

}
