#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#endif

#include <Eigen/Dense>
#include <string>
#include <memory>

#include "RealSymmetric.h"

namespace Maniverse{

inline static Eigen::MatrixXd Symmetrize(Eigen::MatrixXd X){
	return ( X + X.transpose() ) / 2;
}

RealSymmetric::RealSymmetric(Eigen::MatrixXd p, std::string geodesic): Euclidean(p, geodesic){
	this->Name = "RealSymmetric(" + std::to_string(p.rows()) + ", " + std::to_string(p.cols()) + ")";
}

int RealSymmetric::getDimension() const{
	return ( 1 + this->P.rows() ) * this->P.rows() / 2;
}

Eigen::MatrixXd RealSymmetric::TangentProjection(Eigen::MatrixXd A) const{
	return Symmetrize(A);
}

Eigen::MatrixXd RealSymmetric::TangentPurification(Eigen::MatrixXd A) const{
	return Symmetrize(A);
}

void RealSymmetric::setPoint(Eigen::MatrixXd p, bool /*purify*/){
	this->P = Symmetrize(p);
}

void RealSymmetric::getGradient(){
	this->Gr = Symmetrize(this->Ge);
}

Eigen::MatrixXd RealSymmetric::getHessian(Eigen::MatrixXd HeX, Eigen::MatrixXd /*X*/, bool /*weingarten*/) const{
	return Symmetrize(HeX);
}

std::unique_ptr<Manifold> RealSymmetric::Clone() const{
	return std::make_unique<RealSymmetric>(*this);
}

std::shared_ptr<Manifold> RealSymmetric::Share() const{
	return std::make_shared<RealSymmetric>(*this);
}

#ifdef __PYTHON__
void Init_RealSymmetric(pybind11::module_& m){
	pybind11::classh<RealSymmetric, Euclidean>(m, "RealSymmetric")
		.def(pybind11::init<Eigen::MatrixXd, std::string>(), pybind11::arg("p"), pybind11::arg("geodesic") = "EXACT");
}
#endif

}
