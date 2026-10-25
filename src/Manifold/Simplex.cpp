#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#endif

#include <Eigen/Dense>
#include <cmath>
#include <string>
#include <memory>

#include "Simplex.h"

namespace Maniverse{

static double Distance(Eigen::MatrixXd p, Eigen::MatrixXd q){
	return 2 * std::acos( p.cwiseProduct(q).cwiseSqrt().sum() );
}

Simplex::Simplex(Eigen::MatrixXd p, std::string geodesic): Manifold(p, geodesic){
	__Check_Geodesic__("EXACT")
	this->Name = "Simplex(" + std::to_string(p.size()) + ")";
	if ( p.cols() != 1 ) throw std::runtime_error("A point on the Simplex manifold should have only one column!");
}

int Simplex::getDimension() const{
	return this->P.size() - 1;
}

double Simplex::Inner(Eigen::MatrixXd X, Eigen::MatrixXd Y) const{
	return this->P.cwiseInverse().cwiseProduct(X.cwiseProduct(Y)).sum();
}

Eigen::MatrixXd Simplex::Retract(Eigen::MatrixXd X) const{
	if ( X.norm() == 0 ) return this->P;
	const Eigen::MatrixXd Xp = X.cwiseProduct(this->P.array().rsqrt().matrix());
	const double norm = Xp.norm();
	const Eigen::MatrixXd Xpn = Xp / norm;
	const Eigen::MatrixXd tmp1 = 0.5 * (this->P + Xpn.cwiseProduct(Xpn));
	const Eigen::MatrixXd tmp2 = 0.5 * (this->P - Xpn.cwiseProduct(Xpn)) * std::cos(norm);
	const Eigen::MatrixXd tmp3 = Xpn.cwiseProduct(this->P.cwiseSqrt()) * std::sin(norm);
	return tmp1 + tmp2 + tmp3;
}

Eigen::MatrixXd Simplex::InverseRetract(Manifold& N) const{
	__Check_Log_Map__
	const Eigen::MatrixXd q = N.P;
	const double dot = this->P.cwiseSqrt().cwiseProduct(q.cwiseSqrt()).sum();
	const double tmp1 = Distance(this->P, q);
	const double tmp2 = 1. - dot;
	const Eigen::MatrixXd tmp3 = this->P.cwiseProduct(q).cwiseSqrt();
	const Eigen::MatrixXd tmp4 = dot * this->P;
	return tmp1 / tmp2 * ( tmp3 - tmp4 );
}

Eigen::MatrixXd Simplex::TangentProjection(Eigen::MatrixXd A) const{
	return A - this->P * A.sum();
}

Eigen::MatrixXd Simplex::TangentPurification(Eigen::MatrixXd A) const{
	return A.array() - A.mean();
}

void Simplex::setPoint(Eigen::MatrixXd p, bool purify){
	this->P = p;
	if (purify){
		const Eigen::MatrixXd Pabs = this->P.cwiseAbs();
		this->P /= Pabs.sum();
	}
}

void Simplex::getGradient(){
	this->Gr = this->TangentProjection(this->P.cwiseProduct(this->Ge));
}

static Eigen::MatrixXd Projection(Eigen::MatrixXd P, Eigen::MatrixXd A){
	const int n = (int)P.size();
	const Eigen::MatrixXd ones = Eigen::MatrixXd::Zero(n, n).array() + 1;
	Eigen::MatrixXd tmp = Eigen::MatrixXd::Zero(n, n);
	for ( int i = 0; i < n; i++ ) tmp.col(i) = P;
	return ( Eigen::MatrixXd::Identity(n, n) - tmp ) * A;
}

Eigen::MatrixXd Simplex::getHessian(Eigen::MatrixXd HeX, Eigen::MatrixXd X, bool weingarten) const{
	const int n = this->P.size();
	const Eigen::MatrixXd ones = Eigen::MatrixXd::Zero(n, n).array() + 1;
	const Eigen::MatrixXd proj = Projection(this->P, Eigen::MatrixXd::Identity(n, n));
	const Eigen::MatrixXd M = proj * (Eigen::MatrixXd)this->P.asDiagonal();
	const Eigen::MatrixXd N = proj * (Eigen::MatrixXd)(
			this->Ge
			- ones * this->Ge.cwiseProduct(this->P)
			- 0.5 * this->Gr.cwiseProduct(this->P.cwiseInverse())
	).asDiagonal();
	if ( weingarten ) return (Eigen::MatrixXd)(M * HeX + N * X);
	else return (Eigen::MatrixXd)(M * HeX); // Not sure about this one.
}

std::unique_ptr<Manifold> Simplex::Clone() const{
	return std::make_unique<Simplex>(*this);
}

std::shared_ptr<Manifold> Simplex::Share() const{
	return std::make_shared<Simplex>(*this);
}

#ifdef __PYTHON__
void Init_Simplex(pybind11::module_& m){
	pybind11::classh<Simplex, Manifold>(m, "Simplex")
		.def(pybind11::init<Eigen::MatrixXd, std::string>(), pybind11::arg("p"), pybind11::arg("geodesic") = "EXACT");
}
#endif

}
