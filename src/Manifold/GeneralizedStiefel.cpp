#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#endif

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <string>
#include <memory>

#include "GeneralizedStiefel.h"

namespace Maniverse{

// https://doi.org/10.1016/j.cam.2022.114953

GeneralizedStiefel::GeneralizedStiefel(Eigen::MatrixXd p, Eigen::MatrixXd B, std::string geodesic): Manifold(p, geodesic), B(B){
	__Check_Geodesic__("EXACT", "POLAR")
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

int GeneralizedStiefel::getDimension() const{
	const int n = this->P.rows();
	const int k = this->P.cols();
	return n * k - k * ( k + 1 ) / 2;
}

double GeneralizedStiefel::Inner(Eigen::MatrixXd X, Eigen::MatrixXd Y) const{
	return X.cwiseProduct(B * Y).sum();
}

Eigen::MatrixXd GeneralizedStiefel::Flat(Eigen::MatrixXd X) const{
	return B * X;
}

Eigen::MatrixXd GeneralizedStiefel::Sharp(Eigen::MatrixXd X) const{
	return Binv * X;
}

Eigen::MatrixXd GeneralizedStiefel::Retract(Eigen::MatrixXd X) const{
	const int nrows = X.rows();
	const int ncols = X.cols();
	if ( this->Geodesic == "EXACT" ){
		Eigen::MatrixXd A = Eigen::MatrixXd::Zero(nrows, 2 * ncols);
		A << this->P, X;
		Eigen::MatrixXd B = Eigen::MatrixXd::Zero(2 * ncols, 2 * ncols);
		B.topLeftCorner(ncols, ncols) = B.bottomRightCorner(ncols, ncols) = this->P.transpose() * X;
		B.topRightCorner(ncols, ncols) = - X.transpose() * this->B * X;
		B.bottomLeftCorner(ncols, ncols) = Eigen::MatrixXd::Identity(ncols, ncols);
		Eigen::MatrixXd C = Eigen::MatrixXd::Zero(2 * ncols, ncols);
		C.topRows(ncols) = ( - this->P.transpose() * X ).exp();
		return A * B.exp() * C;
	}else if ( this->Geodesic == "POLAR" ){
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(
			Eigen::MatrixXd::Identity(ncols, ncols) + X.transpose() * B * X
		);
		return ( this->P + X ) * es.operatorInverseSqrt();
	}
	__Check_Geodesic_Func__
	return X;
}

inline static Eigen::MatrixXd Sylvester(Eigen::MatrixXd A, Eigen::MatrixXd Q){
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

Eigen::MatrixXd GeneralizedStiefel::InverseRetract(Manifold& N) const{
	__Check_Log_Map__
	const Eigen::MatrixXd p = this->P;
	const Eigen::MatrixXd q = N.P;
	if ( this->Geodesic == "POLAR" ){
		const Eigen::MatrixXd M = p.transpose() * B * q;
		const Eigen::MatrixXd S = Sylvester(M, 2 * Eigen::MatrixXd::Identity(p.cols(), p.cols()));
		return q * S - p;
	}
	__Check_Geodesic_Func__
	return q;
}

Eigen::MatrixXd GeneralizedStiefel::TransportTangent(Eigen::MatrixXd Y, Eigen::MatrixXd Z) const{
	const int nrows = Y.rows();
	const int ncols = Y.cols();
	if ( this->Geodesic == "POLAR" ){
		const Eigen::MatrixXd IplusZtZ = Eigen::MatrixXd::Identity(ncols, ncols) + Z.transpose() * B * Z;
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(IplusZtZ);
		const Eigen::MatrixXd A = es.operatorSqrt();
		const Eigen::MatrixXd Ainv = es.operatorInverseSqrt();
		const Eigen::MatrixXd xiBeta = Y.transpose() * B * Z;
		const Eigen::MatrixXd Q = xiBeta + xiBeta.transpose();
		const Eigen::MatrixXd deri = Sylvester(A, Q);
		return Y * Ainv - ( this->P + Z ) * Ainv * deri * Ainv;
	}
	__Check_Geodesic_Func__
	return Y;
}

Eigen::MatrixXd GeneralizedStiefel::TransportManifold(Eigen::MatrixXd X, Manifold& N) const{
	__Check_Vec_Transport__
	const Eigen::MatrixXd Z = this->InverseRetract(N);
	return this->TransportTangent(X, Z);
}

inline static Eigen::MatrixXd GeneralizedStiefelTangentProjection(Eigen::MatrixXd B, Eigen::MatrixXd P, Eigen::MatrixXd A){
	const Eigen::MatrixXd PtA = P.transpose() * B * A;
	const Eigen::MatrixXd SymPtA = 0.5 * ( PtA + PtA.transpose() );
	return A - P * SymPtA;
}

Eigen::MatrixXd GeneralizedStiefel::TangentProjection(Eigen::MatrixXd X) const{
	return GeneralizedStiefelTangentProjection(B, this->P, X);
}

Eigen::MatrixXd GeneralizedStiefel::TangentPurification(Eigen::MatrixXd X) const{
	return GeneralizedStiefelTangentProjection(B, this->P, X);
}

void GeneralizedStiefel::setPoint(Eigen::MatrixXd p, bool purify){
	if (purify){
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(p.transpose() * B * p);
		p = ( p * es.operatorInverseSqrt() ).eval();
	}
	this->P = p;
}

void GeneralizedStiefel::getGradient(){
	this->Gr = this->TangentProjection(this->Sharp(this->Ge));
}

Eigen::MatrixXd GeneralizedStiefel::getHessian(Eigen::MatrixXd HeX, Eigen::MatrixXd X, bool weingarten) const{
	return GeneralizedStiefelTangentProjection(
		B, this->P,
		Binv * HeX - X * this->P.transpose() * (
			this->Ge - B * this->Gr
		)
	);
}

std::unique_ptr<Manifold> GeneralizedStiefel::Clone() const{
	return std::make_unique<GeneralizedStiefel>(*this);
}

std::shared_ptr<Manifold> GeneralizedStiefel::Share() const{
	return std::make_shared<GeneralizedStiefel>(*this);
}

#ifdef __PYTHON__
void Init_GeneralizedStiefel(pybind11::module_& m){
	pybind11::classh<GeneralizedStiefel, Manifold>(m, "GeneralizedStiefel")
		.def(pybind11::init<Eigen::MatrixXd, Eigen::MatrixXd, std::string>(), pybind11::arg("p"), pybind11::arg("B"), pybind11::arg("geodesic") = "POLAR");
}
#endif

}
