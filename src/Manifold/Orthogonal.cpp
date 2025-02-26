#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#endif
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <functional>
#include <cassert>
#include <typeinfo>
#include <memory>

#include "../Macro.h"

#include "Orthogonal.h"


Orthogonal::Orthogonal(EigenMatrix p, bool matrix_free): Manifold(p, matrix_free){
	this->Name = "Orthogonal";
	if ( p.rows() != p.cols() )
		throw std::runtime_error("An orthogonal matrix must be square!");
	if ( ( p * p.transpose() - p.transpose() * p ).norm() > 1e-8 )
		throw std::runtime_error("An orthogonal matrix must fulfill U.Ut = Ut.U!");
}

int Orthogonal::getDimension() const{
	return this->P.cols() * (this->P.cols() - 1) / 2;
}

double Orthogonal::Inner(EigenMatrix X, EigenMatrix Y) const{
	return 0.5 * Dot(X, Y);
}

EigenMatrix Orthogonal::Exponential(EigenMatrix X) const{
	return (X * this->P.transpose()).exp() * this->P;
}

EigenMatrix Orthogonal::Logarithm(Manifold& N) const{
	__Check_Log_Map__
	const EigenMatrix q = N.P;
	return ( this->P.transpose() * q ).log();
}

EigenMatrix Orthogonal::TangentProjection(EigenMatrix A) const{
	return 0.5 * ( A - this->P * A.transpose() * this->P );
}

EigenMatrix Orthogonal::TangentPurification(EigenMatrix A) const{
	const EigenMatrix Z = this->P.transpose() * A;
	const EigenMatrix Zpurified = 0.5  * (Z - Z.transpose());
	return this->P * Zpurified;
}

void Orthogonal::Update(EigenMatrix p, bool purify){
	this->P = p;
	if (purify){
		Eigen::BDCSVD<EigenMatrix> svd(this->P, Eigen::ComputeFullU | Eigen::ComputeFullV);
		this->P = svd.matrixU() * svd.matrixV().transpose();
	}
}

void Orthogonal::getGradient(){
	this->Gr = this->TangentPurification(this->TangentProjection(this->Ge));
}

void Orthogonal::getHessian(){
	const EigenMatrix P = this->P;
	const EigenMatrix Grp = this->Ge - this->Gr;
	const EigenMatrix PGrpT = this->P * Grp.transpose();
	const std::function<EigenMatrix (EigenMatrix)> He = this->He;
	this->Hr = [P, Grp, PGrpT, He](EigenMatrix v){
		const EigenMatrix Hev = He(v);
		const EigenMatrix TprojHev = 0.5 * ( Hev - P * Hev.transpose() * P );
		return (EigenMatrix)( TprojHev - 0.5 * ( PGrpT * v - P * v.transpose() * Grp ) );
	};
}

std::unique_ptr<Manifold> Orthogonal::Clone() const{
	return std::make_unique<Orthogonal>(*this);
}

#ifdef __PYTHON__
void Init_Orthogonal(pybind11::module_& m){
	pybind11::class_<Orthogonal, Manifold>(m, "Orthogonal")
		.def(pybind11::init<EigenMatrix, bool>());
}
#endif
