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

#include "Simplex.h"


static double Distance(EigenMatrix p, EigenMatrix q){
	return 2 * std::acos( p.cwiseProduct(q).cwiseSqrt().sum() );
}

Simplex::Simplex(EigenMatrix p): Manifold(p){
	this->Name = "Simplex";
	if ( p.cols() != 1 ) throw std::runtime_error("A point on the Simplex manifold should have only one column!");
}

int Simplex::getDimension() const{
	return this->P.size() - 1;
}

double Simplex::Inner(EigenMatrix X, EigenMatrix Y) const{
	return this->P.cwiseInverse().cwiseProduct(X.cwiseProduct(Y)).sum();
}

EigenMatrix Simplex::Exponential(EigenMatrix X) const{
	const EigenMatrix Xp = X.cwiseProduct(this->P.array().rsqrt().matrix());
	const double norm = Xp.norm();
	const EigenMatrix Xpn = Xp / norm;
	const EigenMatrix tmp1 = 0.5 * (this->P + Xpn.cwiseProduct(Xpn));
	const EigenMatrix tmp2 = 0.5 * (this->P - Xpn.cwiseProduct(Xpn)) * std::cos(norm);
	const EigenMatrix tmp3 = Xpn.cwiseProduct(this->P.cwiseSqrt()) * std::sin(norm);
	return tmp1 + tmp2 + tmp3;
}

EigenMatrix Simplex::Logarithm(Manifold& N) const{
	__Check_Log_Map__
	const EigenMatrix q = N.P;
	const double dot = Dot( this->P.cwiseSqrt(), q.cwiseSqrt() );
	const double tmp1 = Distance(this->P, q);
	const double tmp2 = 1. - dot;
	const EigenMatrix tmp3 = this->P.cwiseProduct(q).cwiseSqrt();
	const EigenMatrix tmp4 = dot * this->P;
	return tmp1 / tmp2 * ( tmp3 - tmp4 );
}

EigenMatrix Simplex::TangentProjection(EigenMatrix A) const{
	return A - this->P * A.sum();
}

EigenMatrix Simplex::TangentPurification(EigenMatrix A) const{
	return A.array() - A.mean();
}

void Simplex::setPoint(EigenMatrix p, bool purify){
	this->P = p;
	if (purify){
		const EigenMatrix Pabs = this->P.cwiseAbs();
		this->P /= Pabs.sum();
	}
}

void Simplex::getGradient(){
	this->Gr = this->TangentProjection(this->P.cwiseProduct(this->Ge));
}

static EigenMatrix Projection(EigenMatrix P, EigenMatrix A){
	const int n = (int)P.size();
	const EigenMatrix ones = EigenZero(n, n).array() + 1;
	EigenMatrix tmp = EigenZero(n, n);
	for ( int i = 0; i < n; i++ ) tmp.col(i) = P;
	return ( EigenOne(n, n) - tmp ) * A;
}

std::function<EigenMatrix (EigenMatrix)> Simplex::getHessian(std::function<EigenMatrix (EigenMatrix)> He, bool weingarten) const{
	const int n = this->P.size();
	const EigenMatrix ones = EigenZero(n, n).array() + 1;
	const EigenMatrix proj = Projection(this->P, EigenOne(n, n));
	const EigenMatrix M = proj * (EigenMatrix)this->P.asDiagonal();
	const EigenMatrix N = proj * (EigenMatrix)(
			this->Ge
			- ones * this->Ge.cwiseProduct(this->P)
			- 0.5 * this->Gr.cwiseProduct(this->P.cwiseInverse())
	).asDiagonal();
	if ( weingarten ) return [He, M, N](EigenMatrix v){
		return (EigenMatrix)(M * He(v) + N * v); // The forced conversion "(EigenMatrix)" is necessary. Without it the result will be wrong. I do not know why. Then I forced convert every EigenMatrix return value in std::function for ensurance.
	};
	else return [He, M](EigenMatrix v){ // Not sure about this one.
		return (EigenMatrix)(M * He(v));
	};
}

std::unique_ptr<Manifold> Simplex::Clone() const{
	return std::make_unique<Simplex>(*this);
}

#ifdef __PYTHON__
void Init_Simplex(pybind11::module_& m){
	pybind11::classh<Simplex, Manifold>(m, "Simplex")
		.def(pybind11::init<EigenMatrix>());
}
#endif
