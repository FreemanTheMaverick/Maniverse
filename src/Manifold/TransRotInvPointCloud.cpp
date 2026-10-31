#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#endif

#include <Eigen/Dense>
#include <string>
#include <memory>

#include "TransRotInvPointCloud.h"

namespace Maniverse{

static int getRank(Eigen::MatrixXd p){
	Eigen::FullPivLU<Eigen::MatrixXd> lu(p);
	return lu.rank();
}

TransRotInvPointCloud::TransRotInvPointCloud(Eigen::MatrixXd p, std::string geodesic): Euclidean(p, geodesic){
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

static Eigen::MatrixXd Procrustes(Eigen::MatrixXd P, Eigen::MatrixXd Q, Eigen::MatrixXd X){
	const Eigen::MatrixXd Qinv = Q.completeOrthogonalDecomposition().pseudoInverse();
	Eigen::JacobiSVD<Eigen::MatrixXd, Eigen::ComputeFullU | Eigen::ComputeFullV> svd(Qinv * P);
	const Eigen::MatrixXd Rotation = svd.matrixU() * svd.matrixV().transpose();
	return X * Rotation;
}

static Eigen::MatrixXd Centering(Eigen::MatrixXd Y){
	for ( int i = 0; i < Y.cols(); i++)
		Y.col(i) = ( Y.col(i).array() - Y.col(i).mean() ).matrix();
	return Y;
}

static Eigen::MatrixXd CloudTangentProjection(Eigen::MatrixXd p, Eigen::MatrixXd Y){

	Y = Centering(Y);

	// Y = P Omega
	const int rank = p.cols();
	const int nconstraints = ( rank + 1 ) * rank / 2;
	Eigen::MatrixXd Left = Eigen::MatrixXd::Zero(rank * rank + nconstraints, rank * rank + nconstraints);
	Eigen::VectorXd Right = Eigen::VectorXd::Zero(rank * rank + nconstraints);
	
	// PT * P
	const Eigen::MatrixXd PtP = p.transpose() * p;
	for ( int i = 0; i < rank * rank; i += rank )
		Left.block(i, i, rank, rank) = PtP;

	// Constraints for a vectorized skew-symmetric matrix
	Eigen::MatrixXd C = Eigen::MatrixXd::Zero(nconstraints, rank * rank);
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
	const Eigen::VectorXd x = Left.colPivHouseholderQr().solve(Right);
	const Eigen::MatrixXd Omega = x.head(rank * rank).reshaped(rank, rank);

	// Horizontal component
	return Y - p * Omega;
}


Eigen::MatrixXd TransRotInvPointCloud::InverseRetract(Manifold& N) const{
	__Check_Log_Map__
	return CloudTangentProjection(this->P, N.P);
}

Eigen::MatrixXd TransRotInvPointCloud::TangentProjection(Eigen::MatrixXd A) const{
	return CloudTangentProjection(this->P, A);
}

Eigen::MatrixXd TransRotInvPointCloud::TangentPurification(Eigen::MatrixXd A) const{
	return Centering(A);
}

Eigen::MatrixXd TransRotInvPointCloud::TransportManifold(Eigen::MatrixXd X, Manifold& N) const{
	__Check_Vec_Transport__
	const Eigen::MatrixXd q = N.P;
	const Eigen::MatrixXd rotatedX = Procrustes(q, this->P, X);
	return CloudTangentProjection(q, rotatedX);
}

void TransRotInvPointCloud::setPoint(Eigen::MatrixXd p, bool purify){
	const int rank = getRank(p);
	if ( rank == p.cols() )
		throw std::runtime_error("The matrix is column-rank-deficient!");
	this->P = p;
	if (purify) this->P = this->TangentPurification(p);
}

void TransRotInvPointCloud::getGradient(){
	this->Gr = this->TangentProjection(this->Ge);
}

Eigen::MatrixXd TransRotInvPointCloud::getHessian(Eigen::MatrixXd HeX, Eigen::MatrixXd /*X*/, bool /*weingarten*/) const{
	return this->TangentProjection(HeX);
}

std::unique_ptr<Manifold> TransRotInvPointCloud::Clone() const{
	return std::make_unique<TransRotInvPointCloud>(*this);
}

std::shared_ptr<Manifold> TransRotInvPointCloud::Share() const{
	return std::make_shared<TransRotInvPointCloud>(*this);
}

#ifdef __PYTHON__
void Init_TransRotInvPointCloud(pybind11::module_& m){
	pybind11::classh<TransRotInvPointCloud, Manifold>(m, "TransRotInvPointCloud")
		.def(pybind11::init<Eigen::MatrixXd, std::string>(), pybind11::arg("p"), pybind11::arg("geodesic") = "EXACT");
}
#endif

}
