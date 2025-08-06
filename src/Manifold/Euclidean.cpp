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

Euclidean::Euclidean(EigenMatrix p): Manifold(p){
	this->Name = "Euclidean(" + std::to_string(p.rows()) + ", " + std::to_string(p.cols()) + ")";
}

int Euclidean::getDimension() const{
	return this->P.size();
}

double Euclidean::Inner(EigenMatrix X, EigenMatrix Y) const{
	return (X.cwiseProduct(Y)).sum();
}

EigenMatrix Euclidean::Exponential(EigenMatrix X) const{
	return this->P + X;
}

EigenMatrix Euclidean::Logarithm(Manifold& N) const{
	return N.P - this->P;
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
		.def(pybind11::init<EigenMatrix>());
}
#endif
