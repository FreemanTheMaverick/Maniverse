#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#endif
#include <Eigen/Dense>
#include <typeinfo>
#include <memory>

#include "../Macro.h"

#include "Manifold.h"

Manifold::Manifold(EigenMatrix p){
	this->P.resize(p.rows(), p.cols());
	this->Ge.resize(p.rows(), p.cols());
	this->Gr.resize(p.rows(), p.cols());
	this->P = p;
}

int Manifold::getDimension() const{
	__Not_Implemented__
	return 0;
}

double Manifold::Inner(EigenMatrix X, EigenMatrix Y) const{
	__Not_Implemented__
	return X.rows() * Y.cols() * 0; // Avoiding the unused-variable warning
}

EigenMatrix Manifold::Exponential(EigenMatrix /*X*/) const{
	__Not_Implemented__
	return EigenZero(0, 0);
}

EigenMatrix Manifold::Logarithm(Manifold& /*N*/) const{
	__Not_Implemented__
	return EigenZero(0, 0);
}

EigenMatrix Manifold::TangentProjection(EigenMatrix /*A*/) const{
	__Not_Implemented__
	return EigenZero(0, 0);
}

EigenMatrix Manifold::TangentPurification(EigenMatrix /*A*/) const{
	__Not_Implemented__
	return EigenZero(0, 0);
}

EigenMatrix Manifold::TransportTangent(EigenMatrix /*X*/, EigenMatrix /*Y*/) const{
	__Not_Implemented__
	return EigenZero(0, 0);
}

EigenMatrix Manifold::TransportManifold(EigenMatrix /*X*/, Manifold& /*N*/) const{
	__Not_Implemented__
	return EigenZero(0, 0);
}

void Manifold::setPoint(EigenMatrix /*p*/, bool /*purify*/){
	__Not_Implemented__
}

void Manifold::getGradient(){
	__Not_Implemented__
}

std::function<EigenMatrix (EigenMatrix)> Manifold::getHessian(std::function<EigenMatrix (EigenMatrix)> /*h*/, bool /*weingarten*/) const{
	__Not_Implemented__
	std::function<EigenMatrix (EigenMatrix)> H = [](EigenMatrix){ return EigenZero(0, 0); };
	return H;
}

std::unique_ptr<Manifold> Manifold::Clone() const{
	__Not_Implemented__
	return std::make_unique<Manifold>(*this);
}

#ifdef __PYTHON__
void Init_Manifold(pybind11::module_& m){
	pybind11::classh<Manifold>(m, "Manifold")
		.def_readwrite("Name", &Manifold::Name)
		.def_readwrite("P", &Manifold::P)
		.def_readwrite("Ge", &Manifold::Ge)
		.def_readwrite("Gr", &Manifold::Gr)
		.def(pybind11::init<EigenMatrix>())
		.def("getDimension", &Manifold::getDimension)
		.def("Inner", &Manifold::Inner)
		.def("Exponential", &Manifold::Exponential)
		.def("Logarithm", &Manifold::Logarithm)
		.def("TangentProjection", &Manifold::TangentProjection)
		.def("TangentPurification", &Manifold::TangentPurification)
		.def("TransportTangent", &Manifold::TransportTangent)
		.def("TransportManifold", &Manifold::TransportManifold)
		.def("setPoint", &Manifold::setPoint)
		.def("getGradient", &Manifold::getGradient)
		.def("getHessian", &Manifold::getHessian)
		.def("Clone", &Manifold::Clone);
}
#endif
