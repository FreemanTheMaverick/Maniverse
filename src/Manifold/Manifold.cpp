#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#endif

#include <Eigen/Dense>
#include <string>
#include <memory>

#include "../Macro.h"

#include "Manifold.h"

namespace Maniverse{

Manifold::Manifold(Eigen::MatrixXd p, std::string geodesic){
	this->P.resize(p.rows(), p.cols());
	this->Ge.resize(p.rows(), p.cols());
	this->Gr.resize(p.rows(), p.cols());
	this->P = p;
	for ( char& geodesic_char : geodesic ) geodesic_char = (char)std::toupper(geodesic_char);
	this->Geodesic = geodesic;
}

int Manifold::getDimension() const{
	__Not_Implemented__
	return 0;
}

double Manifold::Inner(Eigen::MatrixXd /*X*/, Eigen::MatrixXd /*Y*/) const{
	__Not_Implemented__
	return 0;
}

Eigen::MatrixXd Manifold::Retract(Eigen::MatrixXd /*X*/) const{
	__Not_Implemented__
	return Eigen::MatrixXd::Zero(0, 0);
}

Eigen::MatrixXd Manifold::InverseRetract(Manifold& /*N*/) const{
	__Not_Implemented__
	return Eigen::MatrixXd::Zero(0, 0);
}

Eigen::MatrixXd Manifold::TangentProjection(Eigen::MatrixXd /*A*/) const{
	__Not_Implemented__
	return Eigen::MatrixXd::Zero(0, 0);
}

Eigen::MatrixXd Manifold::TangentPurification(Eigen::MatrixXd /*A*/) const{
	__Not_Implemented__
	return Eigen::MatrixXd::Zero(0, 0);
}

Eigen::MatrixXd Manifold::TransportTangent(Eigen::MatrixXd /*X*/, Eigen::MatrixXd /*Y*/) const{
	__Not_Implemented__
	return Eigen::MatrixXd::Zero(0, 0);
}

Eigen::MatrixXd Manifold::TransportManifold(Eigen::MatrixXd /*X*/, Manifold& /*N*/) const{
	__Not_Implemented__
	return Eigen::MatrixXd::Zero(0, 0);
}

void Manifold::setPoint(Eigen::MatrixXd /*p*/, bool /*purify*/){
	__Not_Implemented__
}

void Manifold::getGradient(){
	__Not_Implemented__
}

Eigen::MatrixXd Manifold::getHessian(Eigen::MatrixXd HeX, Eigen::MatrixXd /*X*/, bool /*weingarten*/) const{
	__Not_Implemented__
	return HeX;
}

std::unique_ptr<Manifold> Manifold::Clone() const{
	__Not_Implemented__
	return std::make_unique<Manifold>(*this);
}

std::shared_ptr<Manifold> Manifold::Share() const{
	__Not_Implemented__
	return std::make_shared<Manifold>(*this);
}

#ifdef __PYTHON__
class PyManifold : public Manifold, pybind11::trampoline_self_life_support{ public:
	using Manifold::Manifold;

	int getDimension() const override{
		PYBIND11_OVERRIDE(int, Manifold, getDimension);
	}
	double Inner(Eigen::MatrixXd X, Eigen::MatrixXd Y) const override{
		PYBIND11_OVERRIDE(double, Manifold, Inner, X, Y);
	}

	Eigen::MatrixXd Retract(Eigen::MatrixXd X) const override{
		PYBIND11_OVERRIDE(Eigen::MatrixXd, Manifold, Retract, X);
	}
	Eigen::MatrixXd InverseRetract(Manifold& N) const override{
		PYBIND11_OVERRIDE(Eigen::MatrixXd, Manifold, InverseRetract, N);
	}

	Eigen::MatrixXd TangentProjection(Eigen::MatrixXd A) const override{
		PYBIND11_OVERRIDE(Eigen::MatrixXd, Manifold, TangentProjection, A);
	}
	Eigen::MatrixXd TangentPurification(Eigen::MatrixXd A) const override{
		PYBIND11_OVERRIDE(Eigen::MatrixXd, Manifold, TangentPurification, A);
	}

	Eigen::MatrixXd TransportTangent(Eigen::MatrixXd X, Eigen::MatrixXd Y) const override{
		PYBIND11_OVERRIDE(Eigen::MatrixXd, Manifold, TransportTangent, X, Y);
	}
	Eigen::MatrixXd TransportManifold(Eigen::MatrixXd X, Manifold& N) const override{
		PYBIND11_OVERRIDE(Eigen::MatrixXd, Manifold, TransportManifold, X, N);
	}

	void setPoint(Eigen::MatrixXd p, bool purify) override{
		PYBIND11_OVERRIDE(void, Manifold, setPoint, p, purify);
	}

	void getGradient() override{
		PYBIND11_OVERRIDE(void, Manifold, getGradient);
	}
	Eigen::MatrixXd getHessian(Eigen::MatrixXd HeX, Eigen::MatrixXd X, bool weingarten) const override{
		PYBIND11_OVERRIDE(Eigen::MatrixXd, Manifold, getHessian, HeX, X, weingarten);
	}

	std::unique_ptr<Manifold> Clone() const override{
		PYBIND11_OVERRIDE(std::unique_ptr<Manifold>, Manifold, Clone);
	}

	std::shared_ptr<Manifold> Share() const override{
		PYBIND11_OVERRIDE(std::shared_ptr<Manifold>, Manifold, Share);
	}
};

void Init_Manifold(pybind11::module_& m){
	pybind11::classh<Manifold, PyManifold>(m, "Manifold")
		.def_readwrite("Name", &Manifold::Name)
		.def_readwrite("Geodesic", &Manifold::Geodesic)
		.def_readwrite("P", &Manifold::P)
		.def_readwrite("Ge", &Manifold::Ge)
		.def_readwrite("Gr", &Manifold::Gr)
		.def(pybind11::init<Eigen::MatrixXd, std::string>())
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
		.def("Clone", &Manifold::Clone)
		.def("Share", &Manifold::Share);
}
#endif

}
