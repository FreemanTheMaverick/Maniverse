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


static int getRank(EigenMatrix p){
	Eigen::FullPivLU<EigenMatrix> lu(p);
	return lu.rank();
}

static EigenMatrix HorizontalLift(EigenMatrix p, EigenMatrix Y){
	// Y = P Omega
	const int rank = p.cols();
	const int nconstraints = ( rank + 1 ) * rank / 2;
	EigenMatrix Left = EigenZero(rank * rank + nconstraints, rank * rank + nconstraints);
	EigenVector Right = EigenZero(rank * rank + nconstraints);
	
	// PT * P
	const EigenMatrix PtP = p.transpose() * p;
	for ( int i = 0; i < rank * rank; i += rank )
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
	Right.head(rank * rank) = ( p.transpose() * Y ).reshaped(rank * rank, 1);

	// Vertical component
	const EigenVector x = Left.colPivHouseholderQr().solve(Right);
	const EigenMatrix Omega = x.reshaped(rank, rank);

	// Horizontal component
	return Y - p * Omega;
}

TransRotInvPointCloud::TransRotInvPointCloud(EigenMatrix p, bool hess_transport_matrix): Manifold(p, hess_transport_matrix){
	const int rank = getRank(p);
	assert( rank == p.cols() && "The matrix is column-rank-deficient!" );
	this->Name = std::to_string(rank) + "-D translation-rotation-invariant point cloud";
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

EigenMatrix TransRotInvPointCloud::Exponential(EigenMatrix X){
	return X;
}

EigenMatrix TransRotInvPointCloud::Logarithm(EigenMatrix q){
	return q;
}

EigenMatrix TransRotInvPointCloud::TangentProjection(EigenMatrix A){
	EigenMatrix tmp = EigenZero(A.rows(), A.cols());
	for ( int i = 0; i < this->P.cols(); i++)
		tmp.col(i) = A.col(i) / A.col(i).mean();
	return HorizontalLift(this->P, tmp);
}

EigenMatrix TransRotInvPointCloud::TangentPurification(EigenMatrix A){
	EigenMatrix tmp = EigenZero(A.rows(), A.cols());
	for ( int i = 0; i < this->P.cols(); i++)
		tmp.col(i) = A.col(i) / A.col(i).mean();
	return tmp;
}

void TransRotInvPointCloud::Update(EigenMatrix p, bool purify){
	const int rank = getRank(p);
	assert( rank == p.cols() && "The matrix is column-rank-deficient!" );
	this->P = p;
	if (purify)
		this->P = this->TangentPurification(p);
}

void TransRotInvPointCloud::getGradient(){
	this->Gr = this->TangentProjection(this->Ge);
}

void TransRotInvPointCloud::getHessian(){
	const EigenMatrix P = this->P;
	const std::function<EigenMatrix (EigenMatrix)> He = this->He;
	this->Hr = [P, He](EigenMatrix v){
		EigenMatrix tmp = EigenZero(v.rows(), v.cols());
		for ( int i = 0; i < v.cols(); i++)
			tmp.col(i) = v.col(i) / v.col(i).mean();
		return (EigenMatrix)HorizontalLift(P, He( HorizontalLift(P, tmp) ));
	};
}

void Init_TransRotInvPointCloud(pybind11::module_& m){
	pybind11::class_<TransRotInvPointCloud, Manifold>(m, "TransRotInvPointCloud")
		.def(pybind11::init<EigenMatrix, bool>());
}
