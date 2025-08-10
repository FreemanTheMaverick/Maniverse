#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#endif
#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <string>
#include <memory>

#include "../Macro.h"

#include "TransRotInvPointCloud.h"

#include <iostream>


static int getRank(EigenMatrix p){
	Eigen::FullPivLU<EigenMatrix> lu(p);
	return lu.rank();
}

TransRotInvPointCloud::TransRotInvPointCloud(EigenMatrix p): Euclidean(p){
	const int rank = getRank(p);
	if ( rank != p.cols() ) throw std::runtime_error("The matrix is column-rank-deficient!");
	this->Name = "Translation-rotation-invariant-point-cloud(" + std::to_string(p.rows()) + ", " + std::to_string(p.cols()) + ")";
}

int TransRotInvPointCloud::getDimension() const{
	const int nrows = P.rows();
	const int ncols = P.cols();
	//     Total         Trans   Rot
	return nrows * ncols - ncols - ncols * ( ncols - 1 ) / 2;
}

static EigenMatrix Procrustes(EigenMatrix P, EigenMatrix Q, EigenMatrix X){
	Eigen::JacobiSVD<EigenMatrix> svd;
	const EigenMatrix Qinv = Q.completeOrthogonalDecomposition().pseudoInverse();
	svd.compute(Qinv * P, Eigen::ComputeFullU | Eigen::ComputeFullV);
	const EigenMatrix Rotation = svd.matrixU() * svd.matrixV().transpose();
	return X * Rotation;
}

static EigenMatrix Centering(EigenMatrix Y){
	for ( int i = 0; i < Y.cols(); i++)
		Y.col(i) = ( Y.col(i).array() - Y.col(i).mean() ).matrix();
	return Y;
}

static EigenMatrix TangentProjection(EigenMatrix p, EigenMatrix Y){

	Y = Centering(Y);

	// Y = P Omega
	const int rank = p.cols();
	const int nconstraints = ( rank + 1 ) * rank / 2;
	EigenMatrix Left = EigenZero(rank * rank + nconstraints, rank * rank + nconstraints);
	EigenVector Right = EigenZero(rank * rank + nconstraints, 1);
	
	// PT * P
	const EigenMatrix PtP = p.transpose() * p;
	for ( int i = 0; i < rank * rank; i += rank )
		Left.block(i, i, rank, rank) = PtP;

	// Constraints for a vectorized skew-symmetric matrix
	EigenMatrix C = EigenZero(nconstraints, rank * rank);
	int iconstraint = 0;
	for ( int a = 0; a < rank * rank; a += rank + 1, iconstraint++ ){ // Diagonal elements
		C(iconstraint, a) = 1;
	}
	for ( int a = 0; a < rank; a++ ){ // Off-diagonal elements
		for ( int b = 0; b < a; b++, iconstraint++ ){
			C(iconstraint, a * rank + b) = C(iconstraint, b * rank + a) = 1;
		}
	}
	Left.block(0, rank * rank, rank * rank, nconstraints) = C.transpose();
	Left.block(rank * rank, 0, nconstraints, rank * rank) = C;

	// Right-hand side
	Right.head(rank * rank) = ( p.transpose() * Y ).reshaped(rank * rank, 1);

	// Vertical component
	const EigenVector x = Left.colPivHouseholderQr().solve(Right);
	const EigenMatrix Omega = x.head(rank * rank).reshaped(rank, rank);

	// Horizontal component
	return Y - p * Omega;
}


EigenMatrix TransRotInvPointCloud::Logarithm(Manifold& N) const{
	__Check_Log_Map__
	return ::TangentProjection(this->P, N.P);
}

EigenMatrix TransRotInvPointCloud::TangentProjection(EigenMatrix A) const{
	return ::TangentProjection(this->P, A);
}

EigenMatrix TransRotInvPointCloud::TangentPurification(EigenMatrix A) const{
	return Centering(A);
}

EigenMatrix TransRotInvPointCloud::TransportManifold(EigenMatrix X, Manifold& N) const{
	__Check_Vec_Transport__
	const EigenMatrix q = N.P;
	const EigenMatrix rotatedX = Procrustes(q, this->P, X);
	return ::TangentProjection(q, rotatedX);
}

void TransRotInvPointCloud::setPoint(EigenMatrix p, bool purify){
	const int rank = getRank(p);
	if ( rank == p.cols() )
		throw std::runtime_error("The matrix is column-rank-deficient!");
	this->P = p;
	if (purify) this->P = this->TangentPurification(p);
}

void TransRotInvPointCloud::getGradient(){
	this->Gr = this->TangentProjection(this->Ge);
}

std::function<EigenMatrix (EigenMatrix)> TransRotInvPointCloud::getHessian(std::function<EigenMatrix (EigenMatrix)> He, bool /*weingarten*/) const{
	return [P = this->P, He](EigenMatrix v){
		return ::TangentProjection(P, He(v));
	};
}

std::unique_ptr<Manifold> TransRotInvPointCloud::Clone() const{
	return std::make_unique<TransRotInvPointCloud>(*this);
}

#ifdef __PYTHON__
void Init_TransRotInvPointCloud(pybind11::module_& m){
	pybind11::classh<TransRotInvPointCloud, Manifold>(m, "TransRotInvPointCloud")
		.def(pybind11::init<EigenMatrix>());
}
#endif
