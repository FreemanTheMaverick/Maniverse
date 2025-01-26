#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <functional>
#include <cassert>

#include "../Macro.h"

#include "Orthogonal.h"


Orthogonal::Orthogonal(EigenMatrix p, bool hess_transport_matrix): Manifold(p, hess_transport_matrix){
	this->Name = "Orthogonal";
	assert( p.rows() == p.cols() && "An orthogonal matrix must be square!" );
	assert( ( p * p.transpose() - p.transpose() * p ).norm() > 1e-8 && "An orthogonal matrix must fulfill U.Ut = Ut.U!" );
}

int Orthogonal::getDimension(){
	return this->P.cols() * (this->P.cols() - 1) / 2;
}

double Orthogonal::Inner(EigenMatrix X, EigenMatrix Y){
	return 0.5 * Dot(X, Y);
}

EigenMatrix Orthogonal::Exponential(EigenMatrix X){
	return (X * this->P.transpose()).exp() * this->P;
}

EigenMatrix Orthogonal::Logarithm(EigenMatrix q){
	return ( this->P.transpose() * q ).log();
}

EigenMatrix Orthogonal::TangentProjection(EigenMatrix A){
	return 0.5 * ( A - this->P * A.transpose() * this->P );
}

EigenMatrix Orthogonal::TangentPurification(EigenMatrix A){
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

void Init_Orthogonal(pybind11::module_& m){
	pybind11::class_<Orthogonal, Manifold>(m, "Orthogonal")
		.def(pybind11::init<EigenMatrix, bool>());
}
