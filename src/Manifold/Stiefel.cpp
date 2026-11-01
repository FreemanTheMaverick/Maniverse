#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#endif

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <string>
#include <memory>

#include "Stiefel.h"

namespace Maniverse{

// Generalized: https://doi.org/10.1016/j.cam.2022.114953
// https://doi.org/10.1109/TSP.2012.2226167
// https://doi.org/10.1007/s10589-016-9883-4

Stiefel::Stiefel(Eigen::MatrixXd p, std::string geodesic): Manifold(p, geodesic){
	__Check_Geodesic__("EXACT", "POLAR", "QR")
	this->Name = "Stiefel("
		+ std::to_string(p.rows())
		+ ", "
		+ std::to_string(p.cols())
	+ ")";
	if ( ! ( p.transpose() * p ).isApprox(Eigen::MatrixXd::Identity(p.cols(), p.cols())) )
		throw std::runtime_error("A Stiefel matrix must fulfill Ut.U = I!");
}

Stiefel::Stiefel(Eigen::MatrixXd p, Eigen::MatrixXd B, std::string geodesic): Manifold(p, geodesic), B(B){
	__Check_Geodesic__("EXACT", "POLAR", "QR")
	this->Name = "GeneralizedStiefel("
		+ std::to_string(p.rows())
		+ ", "
		+ std::to_string(p.cols())
	+ ")";
	if ( ! ( p.transpose() * B * p ).isApprox(Eigen::MatrixXd::Identity(p.cols(), p.cols())) )
		throw std::runtime_error("A GeneralizedStiefel matrix must fulfill Ut.B.U = I!");
	if ( ! B.transpose().isApprox(B) ) throw std::runtime_error("The B-matrix must be symmetric!");
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(B);
	if ( es.eigenvalues()(0) <= 0 ) throw std::runtime_error("The B-matrix must be positive definite!");
	Binv = es.eigenvectors() * es.eigenvalues().cwiseInverse().asDiagonal() * es.eigenvectors().transpose();
	Bsqrt = es.operatorSqrt();
}

int Stiefel::getDimension() const{
	const int n = this->P.rows();
	const int k = this->P.cols();
	return n * k - k * ( k + 1 ) / 2;
}

double Stiefel::Inner(Eigen::MatrixXd X, Eigen::MatrixXd Y) const{
	return X.cwiseProduct(B.size() ? ( B * Y ).eval() : Y).sum();
}

Eigen::MatrixXd Stiefel::Flat(Eigen::MatrixXd X) const{
	return B.size() ? ( B * X ).eval() : X;
}

Eigen::MatrixXd Stiefel::Sharp(Eigen::MatrixXd X) const{
	return B.size() ? ( Binv * X ).eval() : X;
}

Eigen::MatrixXd Stiefel::Retract(Eigen::MatrixXd X) const{
	const int nrows = X.rows();
	const int ncols = X.cols();
	const Eigen::MatrixXd BX = B.size() ? ( B * X ).eval() : X;
	if ( this->Geodesic == "EXACT" ){
		Eigen::MatrixXd A = Eigen::MatrixXd::Zero(nrows, 2 * ncols);
		A << this->P, X;
		Eigen::MatrixXd B = Eigen::MatrixXd::Zero(2 * ncols, 2 * ncols);
		B.topLeftCorner(ncols, ncols) = B.bottomRightCorner(ncols, ncols) = this->P.transpose() * X;
		B.topRightCorner(ncols, ncols) = - X.transpose() * BX;
		B.bottomLeftCorner(ncols, ncols) = Eigen::MatrixXd::Identity(ncols, ncols);
		Eigen::MatrixXd C = Eigen::MatrixXd::Zero(2 * ncols, ncols);
		C.topRows(ncols) = ( - this->P.transpose() * X ).exp();
		return A * B.exp() * C;
	}else if ( this->Geodesic == "POLAR" ){
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(
			Eigen::MatrixXd::Identity(ncols, ncols) + X.transpose() * BX
		);
		return ( this->P + X ) * es.operatorInverseSqrt();
	}else if ( this->Geodesic == "QR" ){
		// to do
	}
	__Check_Geodesic_Func__
	return X;
}

Eigen::MatrixXd Sylvester(Eigen::MatrixXd A, Eigen::MatrixXd Q){
	// Solve A * X + X * A^t = Q
	// https://discourse.mc-stan.org/t/solve-a-lyapunov-sylvester-equation-include-custom-c-function-using-eigen-library-possible/12688

	const Eigen::MatrixXd B = A.transpose();

	Eigen::ComplexSchur<Eigen::MatrixXd> SchurA(A);
	const Eigen::MatrixXcd R = SchurA.matrixT();
	const Eigen::MatrixXcd U = SchurA.matrixU();

	Eigen::ComplexSchur<Eigen::MatrixXd> SchurB(B);
	const Eigen::MatrixXcd S = SchurB.matrixT();
	const Eigen::MatrixXcd V = SchurB.matrixU();

	const Eigen::MatrixXcd F = U.adjoint() * Q * V;
	const Eigen::MatrixXcd Y = Eigen::internal::matrix_function_solve_triangular_sylvester(R, S, F);
	const Eigen::MatrixXcd X = U * Y * V.adjoint();

	return X.real();
}

Eigen::MatrixXd Stiefel::InverseRetract(Manifold& N) const{
	__Check_Log_Map__
	const Eigen::MatrixXd p = this->P;
	const Eigen::MatrixXd q = N.P;
	const Eigen::MatrixXd Bq = B.size() ? ( B * q ).eval() : q;
	if ( this->Geodesic == "POLAR" ){
		const Eigen::MatrixXd M = p.transpose() * Bq;
		const Eigen::MatrixXd S = Sylvester(M, 2 * Eigen::MatrixXd::Identity(p.cols(), p.cols()));
		return q * S - p;
	}
	__Check_Geodesic_Func__
	return q;
}

Eigen::MatrixXd Stiefel::TransportTangent(Eigen::MatrixXd Y, Eigen::MatrixXd Z) const{
	// Transport Y along Z
	const int nrows = Y.rows();
	const int ncols = Y.cols();
	const Eigen::MatrixXd BZ = B.size() ? ( B * Z ).eval() : Z;
	if ( this->Geodesic == "POLAR" ){
		const Eigen::MatrixXd IplusZtZ = Eigen::MatrixXd::Identity(ncols, ncols) + Z.transpose() * BZ;
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(IplusZtZ);
		const Eigen::MatrixXd A = es.operatorSqrt();
		const Eigen::MatrixXd Ainv = es.operatorInverseSqrt();
		const Eigen::MatrixXd xiBeta = Y.transpose() * BZ;
		const Eigen::MatrixXd Q = xiBeta + xiBeta.transpose();
		const Eigen::MatrixXd deri = Sylvester(A, Q);
		return Y * Ainv - ( this->P + Z ) * Ainv * deri * Ainv;
	}else if ( this->Geodesic == "QR"){
		//todo
	}
	__Check_Geodesic_Func__
	return Y;
}

Eigen::MatrixXd Stiefel::TransportManifold(Eigen::MatrixXd X, Manifold& N) const{
	__Check_Vec_Transport__
	const Eigen::MatrixXd Z = this->InverseRetract(N);
	return this->TransportTangent(X, Z);
}

Eigen::MatrixXd Stiefel::symf(Eigen::MatrixXd PtBX) const{
    return 0.5 * ( PtBX + PtBX.transpose() );
}

Eigen::MatrixXd Stiefel::TangentProjection(Eigen::MatrixXd X) const{
	const Eigen::MatrixXd BX = B.size() ? ( B * X ).eval() : X;
	return X - P * symf(P.transpose() * BX);
}

Eigen::MatrixXd Stiefel::TangentPurification(Eigen::MatrixXd X) const{
	return TangentProjection(X);
}

void Stiefel::setPoint(Eigen::MatrixXd p, bool purify){
	if (purify){
		const Eigen::MatrixXd Bp = B.size() ? ( B * p ).eval() : p;
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(p.transpose() * Bp);
		p = ( p * es.operatorInverseSqrt() ).eval();
	}
	this->P = p;
}

void Stiefel::getGradient(){
	this->Gr = this->TangentProjection(this->Sharp(this->Ge));
}

Eigen::MatrixXd Stiefel::getHessian(Eigen::MatrixXd HeX, Eigen::MatrixXd X) const{
	const Eigen::MatrixXd BinvHeX = B.size() ? ( Binv * HeX ).eval() : HeX;
	const Eigen::MatrixXd BGr = B.size() ? ( B * this->Gr ).eval() : this->Gr;
	return TangentProjection(
		BinvHeX - X * this->P.transpose() * ( this->Ge - BGr )
	);
}

std::unique_ptr<Manifold> Stiefel::Clone() const{
	return std::make_unique<Stiefel>(*this);
}

std::shared_ptr<Manifold> Stiefel::Share() const{
	return std::make_shared<Stiefel>(*this);
}

#ifdef __PYTHON__
void Init_Stiefel(pybind11::module_& m){
	pybind11::classh<Stiefel, Manifold>(m, "Stiefel")
		.def(pybind11::init<Eigen::MatrixXd, std::string>(), pybind11::arg("p"), pybind11::arg("geodesic") = "POLAR")
		.def(pybind11::init<Eigen::MatrixXd, Eigen::MatrixXd, std::string>(), pybind11::arg("p"), pybind11::arg("B"), pybind11::arg("geodesic") = "POLAR");
}
#endif

}
