#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <string>
#include <cassert>

#include "../Macro.h"

#include "TransRotInvPointCloud.h"

#include <iostream>

static int getRank(EigenMatrix p){
	Eigen::FullPivLU<Matrix3f> lu(p);
	return lu.rank();
}

static EigenMatrix HorizontalLift(EigenMatrix p, EigenMatrix Y){
	// Y = P Omega
	const int rank = p.cols();
	const int nconstraints = ( rank + 1 ) * rank / 2;
	const EigenMatrix Left = EigenZero(rank * rank + nconstraints, rank * rank + nconstraints);
	const EigenVector Right = EigenZero(rank * rank + nconstraints);
	
	// PT * P
	const EigenMatrix PtP = p.transpose() * p;
	for ( int i = 0; i < rank; i++ )
		Left.block(i, i, rank, rank) = PtP;

	// Constraints
	EigenMatrix C = EigenZero(nconstraints, rank * rank);
	for ( int a = 0, iconstraint = 0; a < rank; a++ ){
		C(iconstraint, a * rank + a) = 1;
		for ( int b = 0; b < a; b++, iconstraint++ )
			C(iconstraint, a * rank + b) = C(iconstraint, b * rank + a) = 1;
	}
	Left.block(0, rank * rank, rank, nconstraints) = C.transpose();
	Left.block(rank * rank, 0, nconstraints, rank) = C;

	// Right-hand side
	Right.head(rank * rank) = ( p.transpose() * Y ).reshape(rank * rank, 1);

	// Vertical component
	const EigenVector x = Left.colPivHouseholderQr().solve(Right);
	const EigenMatrix Omega = x.reshape(rank, rank);

	// Horizontal component
	return X - p * Omega;
}

TransRotInvPointCloud::TransRotInvPointCloud(EigenMatrix p){
	const int rank = getRank(p);
	assert( rank == p.cols() && "The matrix is column-rank-deficient!" );
	this->Name = std::to_string(rank) + std::to_string("-D translation-rotation-invariant point cloud");
	this->P.resize(p.rows(), p.cols());
	this->Ge.resize(p.rows(), p.cols());
	this->Gr.resize(p.rows(), p.cols());
	this->P = p;
}

int TransRotInvPointCloud::getDimension(){
	const int nrows = P.rows();
	const int ncols = P.cols();
	//     Total         Trans   Rot
	return nrows * ncols - ncols - ncols * ( ncols - 1 ) / 2;
}

double TransRotInvPointCloud::Inner(EigenMatrix X, EigenMatrix Y){
	return (X.cwiseProduct(Y)).sum(); // On the horizontal space
}

std::function<double (EigenMatrix, EigenMatrix)> TransRotInvPointCloud::getInner(){
	const std::function<double (EigenMatrix, EigenMatrix)> inner = [](EigenMatrix X, EigenMatrix Y){
		return (X.cwiseProduct(Y)).sum(); // On the horizontal space
	};
	return inner;
}

double TransRotInvPointCloud::Distance(EigenMatrix q){
	return ( this->P - q ).norm();
}

EigenMatrix TransRotInvPointCloud::Exponential(EigenMatrix X){
	return X;
}

EigenMatrix TransRotInvPointCloud::Logarithm(EigenMatrix q){
	return q;
}

EigenMatrix TransRotInvPointCloud::TangentProjection(EigenMatrix A){
	return HorizontalLift(this->P, A);
}

EigenMatrix TransRotInvPointCloud::TangentPurification(EigenMatrix A){
	return A.array() - A.mean();
}

EigenMatrix TransRotInvPointCloud::TransportTangent(EigenMatrix X, EigenMatrix Y){
	assert( 0 && "Parallel transport on TransRotInvPointCloud manifold is not implemented!" );
	return (X + Y) * 0;
}

EigenMatrix TransRotInvPointCloud::TransportManifold(EigenMatrix X, EigenMatrix q){
	assert( 0 && "Parallel transport on TransRotInvPointCloud manifold is not implemented!" );
	return (X + q) * 0;
}

void TransRotInvPointCloud::Update(EigenMatrix p, bool purify){
	const int rank = getRank(p);
	assert( rank == p.cols() && "The matrix is column-rank-deficient!" );
	this->P = p;
	if (purify){

	}
}

void TransRotInvPointCloud::getGradient(){
	this->Gr = this->TangentProjection(this->P.cwiseProduct(this->Ge));
}

void TransRotInvPointCloud::getHessian(){
	const int n = this->P.size();
	const EigenMatrix ones = EigenZero(n, n).array() + 1;
	const EigenMatrix proj = this->TangentProjection(EigenOne(n, n));
	const EigenMatrix M = proj * (EigenMatrix)this->P.asDiagonal();
	const EigenMatrix N = proj * (EigenMatrix)(
			this->Ge
			- ones * this->Ge.cwiseProduct(this->P)
			- 0.5 * this->Gr.cwiseProduct(this->P.cwiseInverse())
	).asDiagonal();
	const std::function<EigenMatrix (EigenMatrix)> He = this->He;
	this->Hr = [He, M, N](EigenMatrix v){
		return (EigenMatrix)(M * He(v) + N * v); // The forced conversion "(EigenMatrix)" is necessary. Without it the result will be wrong. I do not know why. Then I forced convert every EigenMatrix return value in std::function for ensurance.
	};
}

void Init_TransRotInvPointCloud(pybind11::module_& m){
	pybind11::class_<TransRotInvPointCloud, Manifold>(m, "TransRotInvPointCloud")
		.def(pybind11::init<EigenMatrix>());
}
