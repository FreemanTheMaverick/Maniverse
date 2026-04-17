#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#endif

#include <Eigen/Dense>
#include <cmath>
#include <tuple>
#include <string>
#include <memory>

#include "Grassmann.h"

namespace Maniverse{

Eigen::MatrixXd RealSkewExpm(Eigen::MatrixXd A){
	Eigen::RealSchur<Eigen::MatrixXd> schur(A);
	const Eigen::MatrixXd Q = schur.matrixU();
	const Eigen::MatrixXd T = schur.matrixT();
	const int n = T.cols();
	Eigen::MatrixXd expT = Eigen::MatrixXd::Zero(n, n);
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
	const Eigen::MatrixXd expA = Q * expT * Q.transpose();
	return expA;
}

Grassmann::Grassmann(Eigen::MatrixXd p, std::string geodesic): Manifold(p, geodesic){
	__Check_Geodesic__("EXACT")
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver;
	eigensolver.compute(p);
	const Eigen::VectorXd eigenvalues = eigensolver.eigenvalues();
	const Eigen::MatrixXd eigenvectors = eigensolver.eigenvectors();
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

double Grassmann::Inner(Eigen::MatrixXd X, Eigen::MatrixXd Y) const{
	return X.cwiseProduct(Y).sum();
}

Eigen::MatrixXd Grassmann::Retract(Eigen::MatrixXd X) const{
	const Eigen::MatrixXd Xp = X * this->P - this->P * X;
	const Eigen::MatrixXd pX = - Xp;
	const Eigen::MatrixXd expXp = RealSkewExpm(Xp);
	const Eigen::MatrixXd exppX = RealSkewExpm(pX);
	return expXp * this->P * exppX;
}

Eigen::MatrixXd Grassmann::InverseRetract(Manifold& N) const{
	for ( auto& [cached_NP, cached_Log] : this->LogCache )
		if ( N.P.isApprox(cached_NP) ) return cached_Log;
	__Check_Log_Map__
	Grassmann& N_ = dynamic_cast<Grassmann&>(N);
	const Eigen::MatrixXd U = this->Projector;
	const Eigen::MatrixXd Y = N_.Projector;
	Eigen::JacobiSVD<Eigen::MatrixXd, Eigen::ComputeFullU | Eigen::ComputeFullV> svd;
	svd.compute(Y.transpose() * U);
	const Eigen::MatrixXd Ystar = Y * svd.matrixU() * svd.matrixV().transpose();
	svd.compute( (Eigen::MatrixXd::Identity(U.rows(), U.rows()) - U * U.transpose() ) * Ystar);
	const Eigen::ArrayXd Sigma = svd.singularValues().array().asin();
	Eigen::MatrixXd SIGMA = Eigen::MatrixXd::Zero(U.rows(), U.cols());
	for ( int i = 0; i < Sigma.size(); i++ ) SIGMA(i, i) = Sigma[i];
	const Eigen::MatrixXd Delta = svd.matrixU() * SIGMA * svd.matrixV().transpose();
	const Eigen::MatrixXd Log = Delta * U.transpose() + U * Delta.transpose();
	const Eigen::MatrixXd result = this->TangentPurification(Log);
	this->LogCache.push_back(std::make_tuple(N_.P, result));
	return result;
}

Eigen::MatrixXd Grassmann::TangentProjection(Eigen::MatrixXd X) const{
	// X must be symmetric.
	// https://sites.uclouvain.be/absil/2013.01
	const Eigen::MatrixXd adPX = this->P * X - X * this->P;
	return this->P * adPX - adPX * this->P;
}

Eigen::MatrixXd Grassmann::TangentPurification(Eigen::MatrixXd A) const{
	const Eigen::MatrixXd symA = 0.5 * ( A + A.transpose() );
	const Eigen::MatrixXd pureA = symA - this->P * symA * this->P;
	return 0.5 * ( pureA + pureA.transpose() );
}

Eigen::MatrixXd Grassmann::TransportTangent(Eigen::MatrixXd X, Eigen::MatrixXd Y) const{
	// X - Vector to transport from P
	// Y - Destination on the tangent space of P
	for ( auto& [cached_Y, cached_expdp, cached_exppd]: this->TransportTangentCache )
		if ( Y.isApprox(cached_Y) ) return cached_expdp * X * cached_exppd;
	const Eigen::MatrixXd dp = Y * this->P - this->P * Y;
	const Eigen::MatrixXd pd = - dp;
	const Eigen::MatrixXd expdp = RealSkewExpm(dp);
	const Eigen::MatrixXd exppd = RealSkewExpm(pd);
	this->TransportTangentCache.push_back(std::make_tuple(Y, expdp, exppd));
	return expdp * X * exppd;
}

Eigen::MatrixXd Grassmann::TransportManifold(Eigen::MatrixXd X, Manifold& N) const{
	// X - Vector to transport from P
	__Check_Vec_Transport__
	Grassmann& N_ = dynamic_cast<Grassmann&>(N);
	const Eigen::MatrixXd Y = this->InverseRetract(N_);
	return this->TransportTangent(X, Y);
}

void Grassmann::setPoint(Eigen::MatrixXd p, bool purify){
	this->P = p;
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver;
	eigensolver.compute(p);
	const Eigen::MatrixXd eigenvectors = eigensolver.eigenvectors();
	const int ncols = this->Projector.cols();
	this->Projector = eigenvectors.rightCols(ncols);
	this->LogCache.clear();
	this->TransportTangentCache.clear();
	if (purify) this->P = this->Projector * this->Projector.transpose();
}

void Grassmann::getGradient(){
	this->Gr = this->TangentPurification(this->TangentProjection(this->Ge));
}

Eigen::MatrixXd Grassmann::getHessian(Eigen::MatrixXd HeX, Eigen::MatrixXd X, bool weingarten) const{
	// https://arxiv.org/abs/0709.2205
	const Eigen::MatrixXd PHeX = P * HeX;
	const Eigen::MatrixXd partA = PHeX - PHeX.transpose();
	if ( weingarten ){
		const Eigen::MatrixXd GeX = Ge * X;
		const Eigen::MatrixXd partB = GeX - GeX.transpose();
		const Eigen::MatrixXd sum = partA - partB;
		return (Eigen::MatrixXd)(2 * P * sum);
	}else return (Eigen::MatrixXd)(2 * P * partA);
}

std::unique_ptr<Manifold> Grassmann::Clone() const{
	return std::make_unique<Grassmann>(*this);
}

std::shared_ptr<Manifold> Grassmann::Share() const{
	return std::make_shared<Grassmann>(*this);
}

#ifdef __PYTHON__
void Init_Grassmann(pybind11::module_& m){
	pybind11::classh<Grassmann, Manifold>(m, "Grassmann")
		.def(pybind11::init<Eigen::MatrixXd, std::string>(), pybind11::arg("p"), pybind11::arg("geodesic") = "EXACT");
}
#endif

}
