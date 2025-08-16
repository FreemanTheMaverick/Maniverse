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

Manifold::Manifold(EigenMatrix p, std::string geodesic){
	this->P.resize(p.rows(), p.cols());
	this->Ge.resize(p.rows(), p.cols());
	this->Gr.resize(p.rows(), p.cols());
	this->P = p;
	for ( char& geodesic_char : geodesic ) geodesic_char = (char)std::toupper(geodesic_char);\
	this->Geodesic = geodesic;
}

int Manifold::getDimension() const{
	__Not_Implemented__
	return 0;
}

double Manifold::Inner(EigenMatrix X, EigenMatrix Y) const{
	__Not_Implemented__
	return X.rows() * Y.cols() * 0; // Avoiding the unused-variable warning
}

EigenMatrix Manifold::Retract(EigenMatrix /*X*/) const{
	__Not_Implemented__
	return EigenZero(0, 0);
}

EigenMatrix Manifold::InverseRetract(Manifold& /*N*/) const{
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
class PyManifold : public Manifold, pybind11::trampoline_self_life_support{ public:
	using Manifold::Manifold;

	int getDimension() const override{
		PYBIND11_OVERRIDE(int, Manifold, getDimension,);
	}
	double Inner(EigenMatrix X, EigenMatrix Y) const override{
		PYBIND11_OVERRIDE(double, Manifold, Inner, X, Y);
	}

	EigenMatrix Retract(EigenMatrix X) const override{
		PYBIND11_OVERRIDE(EigenMatrix, Manifold, Retract, X);
	}
	EigenMatrix InverseRetract(Manifold& N) const override{
		PYBIND11_OVERRIDE(EigenMatrix, Manifold, InverseRetract, N);
	}

	EigenMatrix TangentProjection(EigenMatrix A) const override{
		PYBIND11_OVERRIDE(EigenMatrix, Manifold, TangentProjection, A);
	}
	EigenMatrix TangentPurification(EigenMatrix A) const override{
		PYBIND11_OVERRIDE(EigenMatrix, Manifold, TangentPurification, A);
	}

	EigenMatrix TransportTangent(EigenMatrix X, EigenMatrix Y) const override{
		PYBIND11_OVERRIDE(EigenMatrix, Manifold, TransportTangent, X, Y);
	}
	EigenMatrix TransportManifold(EigenMatrix X, Manifold& N) const override{
		PYBIND11_OVERRIDE(EigenMatrix, Manifold, TransportManifold, X, N);
	}

	void setPoint(EigenMatrix p, bool purify) override{
		PYBIND11_OVERRIDE(void, Manifold, setPoint, p, purify);
	}

	void getGradient() override{
		PYBIND11_OVERRIDE(void, Manifold, getGradient,);
	}
	std::function<EigenMatrix (EigenMatrix)> getHessian(std::function<EigenMatrix (EigenMatrix)> He, bool weingarten) const override{
		PYBIND11_OVERRIDE(std::function<EigenMatrix (EigenMatrix)>, Manifold, getHessian, He, weingarten);
	}

	std::unique_ptr<Manifold> Clone() const override{
		PYBIND11_OVERRIDE(std::unique_ptr<Manifold>, Manifold, Clone,);
	}
};

void Init_Manifold(pybind11::module_& m){
	pybind11::classh<Manifold, PyManifold>(m, "Manifold")
		.def_readwrite("Name", &Manifold::Name)
		.def_readwrite("Geodesic", &Manifold::Geodesic)
		.def_readwrite("P", &Manifold::P)
		.def_readwrite("Ge", &Manifold::Ge)
		.def_readwrite("Gr", &Manifold::Gr)
		.def(pybind11::init<EigenMatrix, std::string>())
		.def("getDimension", &Manifold::getDimension)
		.def("Inner", &Manifold::Inner)
		.def("Retract", &Manifold::Retract)
		.def("InverseRetract", &Manifold::InverseRetract)
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
