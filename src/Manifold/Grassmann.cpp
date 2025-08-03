#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#endif
#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <tuple>
#include <memory>

#include "../Macro.h"

#include "Grassmann.h"

EigenMatrix RealSkewExpm(EigenMatrix A){
	Eigen::RealSchur<EigenMatrix> schur(A);
	const EigenMatrix Q = schur.matrixU();
	const EigenMatrix T = schur.matrixT();
	const int n = T.cols();
	EigenMatrix expT = EigenZero(n, n);
	int i = 0;
	while ( i < n ){
		const double a = ( i == n - 1 ) ? 0 : T(i, i + 1);
		if ( i == n - 1 || std::abs(a) < 1e-12 ){
			expT(i, i) = 1;
			i += 1;
		}else{
			const double sina = std::sin(a);
			const double cosa = std::cos(a);
			expT(i, i) = expT(i + 1, i + 1) = cosa;
			expT(i, i + 1) = sina;
			expT(i + 1, i) = - sina;
			i += 2;
		}
	}
	const EigenMatrix expA = Q * expT * Q.transpose();
	return expA;
}

Grassmann::Grassmann(EigenMatrix p): Manifold(p){
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;
	eigensolver.compute(p);
	const EigenVector eigenvalues = eigensolver.eigenvalues();
	const EigenMatrix eigenvectors = eigensolver.eigenvectors();
	int rank = 0;
	for ( int i = 0; i < p.rows(); i++ )
		if ( eigenvalues(i) > 0.5 ) rank++;
	this->Projector.resize(p.rows(), rank);
	this->Projector = eigenvectors.rightCols(rank);
	this->P = this->Projector * this->Projector.transpose();
	this->Name = "Grassmann(" + std::to_string(p.rows()) + ", " + std::to_string(rank) + ")";
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
	const EigenMatrix expXp = RealSkewExpm(Xp);
	const EigenMatrix exppX = RealSkewExpm(pX);
	return expXp * this->P * exppX;
}

EigenMatrix Grassmann::Logarithm(Manifold& N) const{
	for ( auto& [cached_NP, cached_Log] : this->LogCache )
		if ( N.P.isApprox(cached_NP) ) return cached_Log;
	__Check_Log_Map__
	Grassmann& N_ = dynamic_cast<Grassmann&>(N);
	const EigenMatrix U = this->Projector;
	const EigenMatrix Y = N_.Projector;
	Eigen::JacobiSVD<EigenMatrix> svd;
	svd.compute(Y.transpose() * U, Eigen::ComputeFullU | Eigen::ComputeFullV);
	const EigenMatrix Ystar = Y * svd.matrixU() * svd.matrixV().transpose();
	svd.compute( (EigenOne(U.rows(), U.rows()) - U * U.transpose() ) * Ystar);
	const EigenArray Sigma = svd.singularValues().array().asin();
	EigenMatrix SIGMA = EigenZero(U.rows(), U.cols());
	for ( int i = 0; i < Sigma.size(); i++ ) SIGMA(i, i) = Sigma[i];
	const EigenMatrix Delta = svd.matrixU() * SIGMA * svd.matrixV().transpose();
	const EigenMatrix Log = Delta * U.transpose() + U * Delta.transpose();
	const EigenMatrix result = this->TangentPurification(Log);
	this->LogCache.push_back(std::make_tuple(N_.P, result));
	return result;
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
	for ( auto& [cached_Y, cached_expdp, cached_exppd]: this->TransportTangentCache )
		if ( Y.isApprox(cached_Y) ) return cached_expdp * X * cached_exppd;
	const EigenMatrix dp = Y * this->P - this->P * Y;
	const EigenMatrix pd = - dp;
	const EigenMatrix expdp = RealSkewExpm(dp);
	const EigenMatrix exppd = RealSkewExpm(pd);
	this->TransportTangentCache.push_back(std::make_tuple(Y, expdp, exppd));
	return expdp * X * exppd;
}

EigenMatrix Grassmann::TransportManifold(EigenMatrix X, Manifold& N) const{
	// X - Vector to transport from P
	__Check_Vec_Transport__
	Grassmann& N_ = dynamic_cast<Grassmann&>(N);
	const EigenMatrix Y = this->Logarithm(N_);
	return this->TransportTangent(X, Y);
}

void Grassmann::setPoint(EigenMatrix p, bool purify){
	this->P = p;
	Eigen::SelfAdjointEigenSolver<EigenMatrix> eigensolver;
	eigensolver.compute(p);
	const EigenMatrix eigenvectors = eigensolver.eigenvectors();
	const int ncols = this->Projector.cols();
	this->Projector = eigenvectors.rightCols(ncols);
	this->LogCache.clear();
	this->TransportTangentCache.clear();
	if (purify) this->P = this->Projector * this->Projector.transpose();
}

void Grassmann::getGradient(){
	this->Gr = this->TangentPurification(this->TangentProjection(this->Ge));
}

std::function<EigenMatrix (EigenMatrix)> Grassmann::getHessian(std::function<EigenMatrix (EigenMatrix)> He, bool weingarten) const{
	// https://arxiv.org/abs/0709.2205
	if ( weingarten ) return [P = this->P, Ge = this->Ge, He](EigenMatrix v){
		const EigenMatrix Phe = P * He(v);
		const EigenMatrix partA = Phe - Phe.transpose();
		const EigenMatrix Gev = Ge * v;
		const EigenMatrix partB = Gev - Gev.transpose();
		const EigenMatrix sum = partA - partB;
		return (EigenMatrix)(2 * P * sum);
	};
	else return [P = this->P, He](EigenMatrix v){
		const EigenMatrix Phe = P * He(v);
		const EigenMatrix partA = Phe - Phe.transpose();
		return (EigenMatrix)(2 * P * partA);
	};
}

std::unique_ptr<Manifold> Grassmann::Clone() const{
	return std::make_unique<Grassmann>(*this);
}

#ifdef __PYTHON__
void Init_Grassmann(pybind11::module_& m){
	pybind11::classh<Grassmann, Manifold>(m, "Grassmann")
		.def(pybind11::init<EigenMatrix>());
}
#endif
