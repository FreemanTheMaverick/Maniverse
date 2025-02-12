#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <Eigen/Core>
#include <typeinfo>

#include "../Macro.h"

#include "Manifold.h"


#define __Not_Implemented__\
	std::string func_name = __func__;\
	std::string class_name = typeid(this).name();\
	throw std::runtime_error(func_name + " for " + class_name + " is not implemented!");

Manifold::Manifold(EigenMatrix p, bool hess_transport_matrix){
	this->P.resize(p.rows(), p.cols());
	this->Ge.resize(p.rows(), p.cols());
	this->Gr.resize(p.rows(), p.cols());
	this->P = p;
	this->HessTransportMatrix = hess_transport_matrix;
	if (this->HessTransportMatrix){
		this->Hem.resize(p.rows() * p.cols(), p.rows() * p.cols());
		this->Hrm.resize(p.rows() * p.cols(), p.rows() * p.cols());
		this->TransportTangentMatrix(p.rows() * p.cols(), p.rows() * p.cols());
		this->TransportManifoldMatrix(p.rows() * p.cols(), p.rows() * p.cols());
	}else{
		this->Hem.resize(0, 0);
		this->Hrm.resize(0, 0);
		this->TransportTangentMatrix(0, 0);
		this->TransportManifoldMatrix(0, 0);
	}
}

int Manifold::getDimension(){
	__Not_Implemented__
	return 0;
}

double Manifold::Inner(EigenMatrix X, EigenMatrix Y){
	__Not_Implemented__
	return X.rows() * Y.cols() * 0;
}

EigenMatrix Manifold::getGram(){
	const int size = this->P.rows() * this->P.cols();
	EigenMatrix gram = EigenZero(size, size);
	EigenMatrix tmpx = EigenZero(this->P.rows(), this->P.cols());
	EigenMatrix tmpy = EigenZero(this->P.rows(), this->P.cols());
	for ( int jx = 0, x = 0; jx < this->P.cols(); jx++ ) for ( int ix = 0; ix < this->P.rows(); ix++, x++){
		tmpx(ix, jx) = 1;
		for ( int jy = 0, y = 0; jy < this->P.cols(); jy++ ) for ( int iy = 0; iy < this->P.rows() && y <= x; iy++, y++){
			tmpy(iy, jy) = 1;
			gram(x, y) = gram(y, x) = this->Inner(tmpx, tmpy);
			tmpy(iy, jy) = 0;
		}
		tmpx(ix, jx) = 0;
	}
	return gram;
}

EigenMatrix Manifold::Exponential(EigenMatrix X){
	__Not_Implemented__
	return EigenZero(X.rows(), X.cols());
}

EigenMatrix Manifold::Logarithm(EigenMatrix X){
	__Not_Implemented__
	return EigenZero(X.rows(), X.cols());
}

EigenMatrix Manifold::TangentProjection(EigenMatrix A){
	__Not_Implemented__
	return EigenZero(A.rows(), A.cols());
}

EigenMatrix Manifold::TangentPurification(EigenMatrix A){
	__Not_Implemented__
	return EigenZero(A.rows(), A.cols());
}

EigenMatrix Manifold::TransportTangent(EigenMatrix X, EigenMatrix Y){
	__Not_Implemented__
	return EigenZero(X.rows(), Y.cols());
}

EigenMatrix Manifold::TransportManifold(EigenMatrix X, EigenMatrix q){
	__Not_Implemented__
	return EigenZero(X.rows(), q.cols());
}

void Manifold::Update(EigenMatrix p, bool purify){
	if ( purify ? p.rows() : p.cols() ){ // To avoid the unused-variable warning.
		__Not_Implemented__
	}else{
		__Not_Implemented__
	}
}

void Manifold::getGradient(){
	__Not_Implemented__
}

void Manifold::getHessian(){
	__Not_Implemented__
}

void Init_Manifold(pybind11::module_& m){
	pybind11::class_<Manifold>(m, "Manifold")
		.def_readwrite("Name", &Manifold::Name)
		.def_readwrite("P", &Manifold::P)
		.def_readwrite("Ge", &Manifold::Ge)
		.def_readwrite("Gr", &Manifold::Gr)
		.def_readwrite("HessTransportMatrix", &Manifold::HessTransportMatrix)
		.def_readwrite("Hem", &Manifold::Hem)
		.def_readwrite("Hrm", &Manifold::Hrm)
		.def_readwrite("He", &Manifold::He)
		.def_readwrite("Hr", &Manifold::Hr)
		.def(pybind11::init<EigenMatrix, bool>())
		.def("getDimension", &Manifold::getDimension)
		.def("Inner", &Manifold::Inner)
		.def("getGram", &Manifold::getGram)
		.def("Exponential", &Manifold::Exponential)
		.def("Logarithm", &Manifold::Logarithm)
		.def("TangentProjection", &Manifold::TangentProjection)
		.def("TangentPurification", &Manifold::TangentPurification)
		.def_readwrite("TransportTangentMatrix", &Manifold::TransportTangentMatrix)
		.def_readwrite("TransportManifoldMatrix", &Manifold::TransportManifoldMatrix)
		.def("TransportTangent", &Manifold::TransportTangent)
		.def("TransportManifold", &Manifold::TransportManifold)
		.def("Update", &Manifold::Update)
		.def("getGradient", &Manifold::getGradient)
		.def("getHessian", &Manifold::getHessian);
}
