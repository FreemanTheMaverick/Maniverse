#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#endif
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <cmath>
#include <functional>
#include <tuple>
#include <cassert>
#include <memory>

#include "../Macro.h"

#include "Grassmann.h"


Grassmann::Grassmann(EigenMatrix p, bool matrix_free): Manifold(p, matrix_free){
	this->Name = "Grassmann";
	this->P.resize(p.rows(), p.cols());
	this->Ge.resize(p.rows(), p.cols());
	this->Gr.resize(p.rows(), p.cols());
	this->P = p;
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;
	eigensolver.compute(p);
	const EigenVector eigenvalues = eigensolver.eigenvalues();
	const EigenMatrix eigenvectors = eigensolver.eigenvectors();
	int rank = 0;
	for ( int i = 0; i < p.rows(); i++ )
		if ( eigenvalues(i) > 0.5 ) rank++;
	this->Projector.resize(p.rows(), rank);
	this->Projector = eigenvectors.rightCols(rank);
}

int Grassmann::getDimension() const{
	const double rank = this->Projector.cols();
	return rank * ( this->P.rows() - rank );
}

double Grassmann::Inner(EigenMatrix X, EigenMatrix Y) const{
	return Dot(X, Y);
}

EigenMatrix Grassmann::Exponential(EigenMatrix X) const{
	const EigenMatrix Xp = X * this->P - this->P * X;
	const EigenMatrix pX = - Xp;
	const EigenMatrix expXp = Xp.exp();
	const EigenMatrix exppX = pX.exp();
	return expXp * this->P * exppX;
}

EigenMatrix Grassmann::Logarithm(EigenMatrix q) const{
	const EigenMatrix Omega = 0.5 * (
			( EigenOne(q.rows(), q.cols()) - 2 * q ) *
			( EigenOne(q.rows(), q.cols()) - 2 * this->P )
	).log();
	return Omega * this->P - this->P * Omega;
}

EigenMatrix Grassmann::TangentProjection(EigenMatrix X) const{
	// X must be symmetric.
	// https://sites.uclouvain.be/absil/2013.01
	const EigenMatrix adPX = this->P * X - X * this->P;
	return this->P * adPX - adPX * this->P;
}

EigenMatrix Grassmann::TangentPurification(EigenMatrix A) const{
	const EigenMatrix symA = 0.5 * ( A + A.transpose() );
	const EigenMatrix pureA = symA - this->P * symA * this->P;
	return 0.5 * ( pureA + pureA.transpose() );
}

EigenMatrix Grassmann::TransportTangent(EigenMatrix X, EigenMatrix Y) const{
	// X - Vector to transport from P
	// Y - Destination on the tangent space of P
	const EigenMatrix dp = Y * this->P - this->P * Y;
	const EigenMatrix pd = - dp;
	const EigenMatrix expdp = dp.exp();
	const EigenMatrix exppd = pd.exp();
	return expdp * X * exppd;
}

EigenMatrix Grassmann::TransportManifold(EigenMatrix X, EigenMatrix q) const{
	// X - Vector to transport from P
	// q - Destination on the manifold
	const EigenMatrix Y = this->Logarithm(q);
	return this->TransportTangent(X, Y);
}

void Grassmann::Update(EigenMatrix p, bool purify){
	this->P = p;
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;
	eigensolver.compute(p);
	const EigenMatrix eigenvectors = eigensolver.eigenvectors();
	const int ncols = this->Projector.cols();
	this->Projector = eigenvectors.rightCols(ncols);
	if (purify) this->P = this->Projector * this->Projector.transpose();
}

void Grassmann::getGradient(){
	this->Gr = this->TangentProjection(this->Ge);
}

void Grassmann::getHessian(){
	// https://arxiv.org/abs/0709.2205
	this->Hr = [P = this->P, Ge = this->Ge, He = this->He](EigenMatrix v){
		const EigenMatrix he = He(v);
		const EigenMatrix partA = P * he - he * P;
		const EigenMatrix partB = Ge * v - v * Ge;
		const EigenMatrix sum = partA - partB;
		return (EigenMatrix)(P * sum - sum * P);
	};
}

std::unique_ptr<Manifold> Grassmann::Clone() const{
	return std::make_unique<Grassmann>(*this);
}

#ifdef __PYTHON__
void Init_Grassmann(pybind11::module_& m){
	pybind11::class_<Grassmann, Manifold>(m, "Grassmann")
		.def(pybind11::init<EigenMatrix, bool>());
}
#endif
