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


static EigenMatrix Sylvester(EigenMatrix A, EigenMatrix Q){
	// https://discourse.mc-stan.org/t/solve-a-lyapunov-sylvester-equation-include-custom-c-function-using-eigen-library-possible/12688

	const EigenMatrix B = A.transpose();

	Eigen::ComplexSchur<EigenMatrix> SchurA(A);
	const Eigen::MatrixXcd R = SchurA.matrixT();
	const Eigen::MatrixXcd U = SchurA.matrixU();

	Eigen::ComplexSchur<EigenMatrix> SchurB(B);
	const Eigen::MatrixXcd S = SchurB.matrixT();
	const Eigen::MatrixXcd V = SchurB.matrixU();

	const Eigen::MatrixXcd F = U.adjoint() * Q * V;
	const Eigen::MatrixXcd Y = Eigen::internal::matrix_function_solve_triangular_sylvester(R, S, F);
	const Eigen::MatrixXcd X = U * Y * V.adjoint();

	return X.real();
}

static EigenMatrix OrthPolarRetr(EigenMatrix p, EigenMatrix X){
	Eigen::BDCSVD<EigenMatrix> svd;
	svd.compute(p + X, Eigen::ComputeFullU | Eigen::ComputeFullV);
	return svd.matrixU() * svd.matrixV().transpose();
}

static EigenMatrix OrthPolarInvRetr(EigenMatrix p, EigenMatrix q){
	// Algorithm 2, https://doi.org/10.1109/TSP.2012.2226167
	const EigenMatrix M = p.transpose() * q;
	const EigenMatrix S = Sylvester(M, 2 * EigenOne(p.cols(), p.cols()));
	return q * S - p;
}


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
	return Dot(X, Y);
}

EigenMatrix Orthogonal::Exponential(EigenMatrix X) const{
	return OrthPolarRetr(this->P, X);
	//return (X * this->P.transpose()).exp() * this->P;
}

EigenMatrix Orthogonal::Logarithm(Manifold& N) const{
	__Check_Log_Map__
	const EigenMatrix q = N.P;
	return ( this->P.transpose() * q ).log();
}

EigenMatrix Orthogonal::TangentProjection(EigenMatrix A) const{
	//https://juliamanifolds.github.io/Manifolds.jl/stable/manifolds/stiefel
	const EigenMatrix PtA = this->P.transpose() * A;
	const EigenMatrix symPtA = ( PtA + PtA.transpose() ) / 2;
	const EigenMatrix projA = A - this->P * symPtA;
	return projA;
}

EigenMatrix Orthogonal::TangentPurification(EigenMatrix A) const{
	const EigenMatrix Z = this->P.transpose() * A;
	const EigenMatrix Zpurified = 0.5  * (Z - Z.transpose());
	return this->P * Zpurified;
}

EigenMatrix Orthogonal::TransportTangent(EigenMatrix Y, EigenMatrix Z) const{
	// Transport Y along Z
	// Section 3.5, https://doi.org/10.1007/s10589-016-9883-4
	const EigenMatrix IplusZtZ = EigenOne(Z.cols(), Z.cols()) + Z.transpose() * Z;
	Eigen::SelfAdjointEigenSolver<EigenMatrix> es(IplusZtZ);
	const EigenMatrix A = es.operatorSqrt();
	const EigenMatrix Ainv = es.operatorInverseSqrt();
	const EigenMatrix RZ = OrthPolarRetr(this->P, Z);
	const EigenMatrix RZtY = RZ.transpose() * Y;
	const EigenMatrix Q = RZtY - RZtY.transpose();
	const EigenMatrix Lambda = Sylvester(A, Q);
	return RZ * Lambda + ( EigenOne(Z.cols(), Z.cols()) - RZ * RZ.transpose() ) * Y * Ainv;
}

EigenMatrix Orthogonal::TransportManifold(EigenMatrix X, Manifold& N) const{
	__Check_Vec_Transport__
	const EigenMatrix q = N.P;
	const EigenMatrix Z = OrthPolarInvRetr(this->P, q);
	return this->TransportTangent(X, Z);
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
	//https://juliamanifolds.github.io/Manifolds.jl/stable/manifolds/stiefel
	const EigenMatrix P = this->P;
	const EigenMatrix tmp = this->Ge.transpose() * this->P + this->P.transpose() * this->Ge;
	const std::function<EigenMatrix (EigenMatrix)> He = this->He;
	this->Hr = [P, tmp, He](EigenMatrix v){
		const EigenMatrix A = He(v) - 0.5 * v * tmp;
		const EigenMatrix PtA = P.transpose() * A;
		const EigenMatrix symPtA = ( PtA + PtA.transpose() ) / 2;
		const EigenMatrix projA = A - P * symPtA;
		return (EigenMatrix)(projA);
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
