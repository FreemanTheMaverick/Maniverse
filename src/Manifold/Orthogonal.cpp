#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#endif

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <string>
#include <memory>

#include "Orthogonal.h"

namespace Maniverse{

Orthogonal::Orthogonal(Eigen::MatrixXd p, std::string geodesic): Stiefel(p, geodesic){
	this->Name = "Orthogonal(" + std::to_string(p.rows()) + ", " + std::to_string(p.cols()) + ")";
	if ( p.rows() != p.cols() )
		throw std::runtime_error("An orthogonal matrix must be square!");
}

Eigen::MatrixXd Orthogonal::Retract(Eigen::MatrixXd X) const{
	if ( this->Geodesic == "EXACT" ) return (X * this->P.transpose()).exp() * this->P;
	else if ( this->Geodesic == "POLAR" ) return Stiefel::Retract(X);
	__Check_Geodesic_Func__
	return X;
}

Eigen::MatrixXd Orthogonal::InverseRetract(Manifold& N) const{
	__Check_Log_Map__
	const Eigen::MatrixXd p = this->P;
	const Eigen::MatrixXd q = N.P;
	if ( this->Geodesic == "EXACT" ) return ( q * p.transpose() ).log() * p;
	else if ( this->Geodesic == "POLAR" ) return Stiefel::InverseRetract(N);
	__Check_Geodesic_Func__
	return q;
}

std::unique_ptr<Manifold> Orthogonal::Clone() const{
	return std::make_unique<Orthogonal>(*this);
}

std::shared_ptr<Manifold> Orthogonal::Share() const{
	return std::make_shared<Orthogonal>(*this);
}

#ifdef __PYTHON__
void Init_Orthogonal(pybind11::module_& m){
	pybind11::classh<Orthogonal, Stiefel>(m, "Orthogonal")
		.def(pybind11::init<Eigen::MatrixXd, std::string>(), pybind11::arg("p"), pybind11::arg("geodesic") = "POLAR");
}
#endif

}
