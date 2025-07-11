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

#include "Euclidean.h"


static double Distance(EigenMatrix p, EigenMatrix q){
	return 2 * std::acos( p.cwiseProduct(q).cwiseSqrt().sum() );
}

Euclidean::Euclidean(EigenMatrix p, bool matrix_free): Manifold(p, matrix_free){
	this->Name = "Euclidean";
}

int Euclidean::getDimension() const{
	return this->P.size();
}

double Euclidean::Inner(EigenMatrix X, EigenMatrix Y) const{
	return (X.cwiseProduct(Y)).sum();
}

EigenMatrix Euclidean::Exponential(EigenMatrix X) const{
	return X;
}

EigenMatrix Euclidean::Logarithm(Manifold& N) const{
	return N.P;
}

EigenMatrix Euclidean::TangentProjection(EigenMatrix A) const{
	return A;
}

EigenMatrix Euclidean::TangentPurification(EigenMatrix A) const{
	return A;
}

void Euclidean::Update(EigenMatrix p, [[maybe_unused]] bool purify){
	this->P = p;
}

void Euclidean::getGradient(){
	this->Gr = this->Ge;
}

void Euclidean::getHessian(){
	const std::function<EigenMatrix (EigenMatrix)> He = this->He;
	this->Hr = [He](EigenMatrix v){
		return (EigenMatrix)(He(v));
	};
}

std::unique_ptr<Manifold> Euclidean::Clone() const{
	return std::make_unique<Euclidean>(*this);
}

#ifdef __PYTHON__
void Init_Euclidean(pybind11::module_& m){
	pybind11::class_<Euclidean, Manifold>(m, "Euclidean")
		.def(pybind11::init<EigenMatrix, bool>());
}
#endif
