#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#endif

#include <Eigen/Dense>
#include <string>
#include <memory>

#include "Euclidean.h"

namespace Maniverse{

Euclidean::Euclidean(Eigen::MatrixXd p, std::string geodesic): Manifold(p, geodesic){
	__Check_Geodesic__("EXACT")
	this->Name = "Euclidean(" + std::to_string(p.rows()) + ", " + std::to_string(p.cols()) + ")";
}

int Euclidean::getDimension() const{
	return this->P.size();
}

double Euclidean::Inner(Eigen::MatrixXd X, Eigen::MatrixXd Y) const{
	return X.cwiseProduct(Y).sum();
}

Eigen::MatrixXd Euclidean::Retract(Eigen::MatrixXd X) const{
	return this->P + X;
}

Eigen::MatrixXd Euclidean::InverseRetract(Manifold& N) const{
	__Check_Log_Map__
	return N.P - this->P;
}

Eigen::MatrixXd Euclidean::TransportTangent(Eigen::MatrixXd X, Eigen::MatrixXd /*Y*/) const{
	return X;
}

Eigen::MatrixXd Euclidean::TransportManifold(Eigen::MatrixXd X, Manifold& N) const{
	__Check_Log_Map__
	return X;
}

Eigen::MatrixXd Euclidean::TangentProjection(Eigen::MatrixXd A) const{
	return A;
}

Eigen::MatrixXd Euclidean::TangentPurification(Eigen::MatrixXd A) const{
	return A;
}

void Euclidean::setPoint(Eigen::MatrixXd p, bool /*purify*/){
	this->P = p;
}

void Euclidean::getGradient(){
	this->Gr = this->Ge;
}

Eigen::MatrixXd Euclidean::getHessian(Eigen::MatrixXd HeX, Eigen::MatrixXd /*X*/, bool /*weingarten*/) const{
	return HeX;
}

std::unique_ptr<Manifold> Euclidean::Clone() const{
	return std::make_unique<Euclidean>(*this);
}

std::shared_ptr<Manifold> Euclidean::Share() const{
	return std::make_shared<Euclidean>(*this);
}

#ifdef __PYTHON__
void Init_Euclidean(pybind11::module_& m){
	pybind11::classh<Euclidean, Manifold>(m, "Euclidean")
		.def(pybind11::init<Eigen::MatrixXd, std::string>(), pybind11::arg("p"), pybind11::arg("geodesic") = "EXACT");
}
#endif

}
