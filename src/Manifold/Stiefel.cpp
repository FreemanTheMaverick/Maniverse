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

int Stiefel::getDimension() const{
	const int n = this->P.rows();
	const int k = this->P.cols();
	return n * k - k * ( k + 1 ) / 2;
}

double Stiefel::Inner(Eigen::MatrixXd X, Eigen::MatrixXd Y) const{
	return X.cwiseProduct(Y).sum();
}

Eigen::MatrixXd Stiefel::Retract(Eigen::MatrixXd X) const{
	const int nrows = X.rows();
	const int ncols = X.cols();
	if ( this->Geodesic == "EXACT" ){
		Eigen::MatrixXd A = Eigen::MatrixXd::Zero(nrows, 2 * ncols);
		A << this->P, X;
		Eigen::MatrixXd B = Eigen::MatrixXd::Zero(2 * ncols, 2 * ncols);
		B.topLeftCorner(ncols, ncols) = B.bottomRightCorner(ncols, ncols) = this->P.transpose() * X;
		B.topRightCorner(ncols, ncols) = - X.transpose() * X;
		B.bottomLeftCorner(ncols, ncols) = Eigen::MatrixXd::Identity(ncols, ncols);
		Eigen::MatrixXd C = Eigen::MatrixXd::Zero(2 * ncols, ncols);
		C.topRows(ncols) = ( - this->P.transpose() * X ).exp();
		return A * B.exp() * C;
	}else if ( this->Geodesic == "POLAR" ){
		Eigen::BDCSVD<Eigen::MatrixXd, Eigen::ComputeThinU | Eigen::ComputeFullV> svd(this->P + X);
		return svd.matrixU() * svd.matrixV().transpose();
	}else if ( this->Geodesic == "QR" ){
		Eigen::HouseholderQR<Eigen::MatrixXd> qr(this->P + X);
		return qr.householderQ() * Eigen::MatrixXd::Identity(nrows, ncols);
	}
	__Check_Geodesic_Func__
	return X;
}

inline static Eigen::MatrixXd Sylvester(Eigen::MatrixXd A, Eigen::MatrixXd Q){
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
	// https://doi.org/10.1109/TSP.2012.2226167
	__Check_Log_Map__
	const Eigen::MatrixXd p = this->P;
	const Eigen::MatrixXd q = N.P;
	if ( this->Geodesic == "POLAR" ){ // Algorithm 2
		const Eigen::MatrixXd M = p.transpose() * q;
		const Eigen::MatrixXd S = Sylvester(M, 2 * Eigen::MatrixXd::Identity(p.cols(), p.cols()));
		return q * S - p;
	}
	__Check_Geodesic_Func__
	return q;
}

Eigen::MatrixXd Stiefel::TransportTangent(Eigen::MatrixXd Y, Eigen::MatrixXd Z) const{
	// Transport Y along Z
	// Section 3.5, https://doi.org/10.1007/s10589-016-9883-4
	const int nrows = Y.rows();
	const int ncols = Y.cols();
	if ( this->Geodesic == "POLAR" ){
		const Eigen::MatrixXd IplusZtZ = Eigen::MatrixXd::Identity(ncols, ncols) + Z.transpose() * Z;
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(IplusZtZ);
		const Eigen::MatrixXd A = es.operatorSqrt();
		const Eigen::MatrixXd Ainv = es.operatorInverseSqrt();
		const Eigen::MatrixXd RZ = this->Retract(Z);
		const Eigen::MatrixXd RZtY = RZ.transpose() * Y;
		const Eigen::MatrixXd Q = RZtY - RZtY.transpose();
		const Eigen::MatrixXd Lambda = Sylvester(A, Q);
		return RZ * Lambda + ( Eigen::MatrixXd::Identity(nrows, nrows) - RZ * RZ.transpose() ) * Y * Ainv;
	}else if ( this->Geodesic == "QR" ){
		Eigen::HouseholderQR<Eigen::MatrixXd> qr(this->P + Z);
		const Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(nrows, ncols);
		const Eigen::MatrixXd Rinv = qr.matrixQR().topLeftCorner(ncols, ncols).triangularView<Eigen::Upper>();
		Eigen::MatrixXd TMP = Q.transpose() * Y * Rinv;
		for ( int i = 0; i < ncols; i++ ){
			for ( int j = 0; j < ncols; j++ ){
				if ( i == j ) TMP(i, j) = 0;
				else if ( i < j ) TMP(i, j) = - TMP(i, j);
			}
		}
		return Q * TMP + ( Eigen::MatrixXd::Identity(nrows, nrows) - Q * Q.transpose() ) * Y * Rinv;
	}
	__Check_Geodesic_Func__
	return Y;
}

Eigen::MatrixXd Stiefel::TransportManifold(Eigen::MatrixXd X, Manifold& N) const{
	__Check_Vec_Transport__
	const Eigen::MatrixXd Z = this->InverseRetract(N);
	return this->TransportTangent(X, Z);
}

inline static Eigen::MatrixXd StiefelTangentProjection(Eigen::MatrixXd P, Eigen::MatrixXd A){
	//https://juliamanifolds.github.io/Manifolds.jl/stable/manifolds/stiefel
	const Eigen::MatrixXd PtA = P.transpose() * A;
	const Eigen::MatrixXd SymPtA = 0.5 * ( PtA + PtA.transpose() );
	return A - P * SymPtA;
}

Eigen::MatrixXd Stiefel::TangentProjection(Eigen::MatrixXd X) const{
	return StiefelTangentProjection(this->P, X);
}

Eigen::MatrixXd Stiefel::TangentPurification(Eigen::MatrixXd X) const{
	return StiefelTangentProjection(this->P, X);
}

void Stiefel::setPoint(Eigen::MatrixXd p, bool purify){
	if (purify){
		Eigen::BDCSVD<Eigen::MatrixXd, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(p);
		p = svd.matrixU() * svd.matrixV().transpose();
	}
	this->P = p;
}

void Stiefel::getGradient(){
	this->Gr = this->TangentProjection(this->Ge);
}

Eigen::MatrixXd Stiefel::getHessian(Eigen::MatrixXd HeX, Eigen::MatrixXd X, bool weingarten) const{
	//https://juliamanifolds.github.io/Manifolds.jl/stable/manifolds/stiefel
	if ( ! weingarten ) return StiefelTangentProjection(this->P, HeX);
	else{
		const Eigen::MatrixXd tmp = this->Ge.transpose() * this->P + this->P.transpose() * this->Ge;
		return StiefelTangentProjection(this->P, HeX - 0.5 * X * tmp);
	}
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
		.def(pybind11::init<Eigen::MatrixXd, std::string>(), pybind11::arg("p"), pybind11::arg("geodesic") = "POLAR");
}
#endif

}
