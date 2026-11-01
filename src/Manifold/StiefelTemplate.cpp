#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#endif

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <memory>
#include <stdexcept>
#include <string>

#include "StiefelTemplate.h"

namespace Maniverse{

namespace{

Eigen::MatrixXd StiefelTemplateSylvester(Eigen::MatrixXd A, Eigen::MatrixXd Q){
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

}

template<bool IsGeneralized>
StiefelTemplate<IsGeneralized>::StiefelTemplate(Eigen::MatrixXd p, std::string geodesic): Manifold(p, geodesic){
	if constexpr (IsGeneralized){
		throw std::invalid_argument("A generalized Stiefel template requires a B-matrix!");
	}else{
		__Check_Geodesic__("EXACT", "POLAR", "QR")
		this->Name = "Stiefel("
			+ std::to_string(p.rows())
			+ ", "
			+ std::to_string(p.cols())
			+ ")";
		if ( ! ( p.transpose() * p ).isApprox(Eigen::MatrixXd::Identity(p.cols(), p.cols())) )
			throw std::runtime_error("A Stiefel matrix must fulfill Ut.U = I!");
	}
}

template<bool IsGeneralized>
StiefelTemplate<IsGeneralized>::StiefelTemplate(Eigen::MatrixXd p, Eigen::MatrixXd B, std::string geodesic): Manifold(p, geodesic){
	if constexpr (!IsGeneralized){
		throw std::invalid_argument("A Stiefel template does not accept a B-matrix!");
	}else{
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
		this->B = B;
		this->Binv = es.eigenvectors() * es.eigenvalues().cwiseInverse().asDiagonal() * es.eigenvectors().transpose();
		this->Bsqrt = es.operatorSqrt();
	}
}

template<bool IsGeneralized>
int StiefelTemplate<IsGeneralized>::getDimension() const{
	const int n = this->P.rows();
	const int k = this->P.cols();
	return n * k - k * ( k + 1 ) / 2;
}

template<bool IsGeneralized>
double StiefelTemplate<IsGeneralized>::Inner(Eigen::MatrixXd X, Eigen::MatrixXd Y) const{
	if constexpr (IsGeneralized) return X.cwiseProduct(this->B * Y).sum();
	return X.cwiseProduct(Y).sum();
}

template<bool IsGeneralized>
Eigen::MatrixXd StiefelTemplate<IsGeneralized>::Flat(Eigen::MatrixXd X) const{
	if constexpr (IsGeneralized) return this->B * X;
	return X;
}

template<bool IsGeneralized>
Eigen::MatrixXd StiefelTemplate<IsGeneralized>::Sharp(Eigen::MatrixXd X) const{
	if constexpr (IsGeneralized) return this->Binv * X;
	return X;
}

template<bool IsGeneralized>
Eigen::MatrixXd StiefelTemplate<IsGeneralized>::Retract(Eigen::MatrixXd X) const{
	const int nrows = X.rows();
	const int ncols = X.cols();
	if ( this->Geodesic == "EXACT" ){
		Eigen::MatrixXd A = Eigen::MatrixXd::Zero(nrows, 2 * ncols);
		A << this->P, X;
		Eigen::MatrixXd Block = Eigen::MatrixXd::Zero(2 * ncols, 2 * ncols);
		Block.topLeftCorner(ncols, ncols) = Block.bottomRightCorner(ncols, ncols) = this->P.transpose() * X;
		if constexpr (IsGeneralized) Block.topRightCorner(ncols, ncols) = - X.transpose() * this->B * X;
		else Block.topRightCorner(ncols, ncols) = - X.transpose() * X;
		Block.bottomLeftCorner(ncols, ncols) = Eigen::MatrixXd::Identity(ncols, ncols);
		Eigen::MatrixXd C = Eigen::MatrixXd::Zero(2 * ncols, ncols);
		C.topRows(ncols) = ( - this->P.transpose() * X ).exp();
		return A * Block.exp() * C;
	}else if ( this->Geodesic == "POLAR" ){
		if constexpr (IsGeneralized){
			Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(
				Eigen::MatrixXd::Identity(ncols, ncols) + X.transpose() * this->B * X
			);
			return ( this->P + X ) * es.operatorInverseSqrt();
		}else{
			Eigen::BDCSVD<Eigen::MatrixXd, Eigen::ComputeThinU | Eigen::ComputeFullV> svd(this->P + X);
			return svd.matrixU() * svd.matrixV().transpose();
		}
	}else if ( this->Geodesic == "QR" ){
		if constexpr (!IsGeneralized){
			Eigen::HouseholderQR<Eigen::MatrixXd> qr(this->P + X);
			return qr.householderQ() * Eigen::MatrixXd::Identity(nrows, ncols);
		}
	}
	__Check_Geodesic_Func__
	return X;
}

template<bool IsGeneralized>
Eigen::MatrixXd StiefelTemplate<IsGeneralized>::InverseRetract(Manifold& N) const{
	__Check_Log_Map__
	const Eigen::MatrixXd p = this->P;
	const Eigen::MatrixXd q = N.P;
	if ( this->Geodesic == "POLAR" ){
		Eigen::MatrixXd M;
		if constexpr (IsGeneralized) M = p.transpose() * this->B * q;
		else M = p.transpose() * q;
		const Eigen::MatrixXd S = StiefelTemplateSylvester(M, 2 * Eigen::MatrixXd::Identity(p.cols(), p.cols()));
		return q * S - p;
	}
	__Check_Geodesic_Func__
	return q;
}

template<bool IsGeneralized>
Eigen::MatrixXd StiefelTemplate<IsGeneralized>::TransportTangent(Eigen::MatrixXd Y, Eigen::MatrixXd Z) const{
	const int nrows = Y.rows();
	const int ncols = Y.cols();
	if ( this->Geodesic == "POLAR" ){
		if constexpr (IsGeneralized){
			const Eigen::MatrixXd IplusZtZ = Eigen::MatrixXd::Identity(ncols, ncols) + Z.transpose() * this->B * Z;
			Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(IplusZtZ);
			const Eigen::MatrixXd A = es.operatorSqrt();
			const Eigen::MatrixXd Ainv = es.operatorInverseSqrt();
			const Eigen::MatrixXd xiBeta = Y.transpose() * this->B * Z;
			const Eigen::MatrixXd Q = xiBeta + xiBeta.transpose();
			const Eigen::MatrixXd deri = StiefelTemplateSylvester(A, Q);
			return Y * Ainv - ( this->P + Z ) * Ainv * deri * Ainv;
		}else{
			const Eigen::MatrixXd IplusZtZ = Eigen::MatrixXd::Identity(ncols, ncols) + Z.transpose() * Z;
			Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(IplusZtZ);
			const Eigen::MatrixXd A = es.operatorSqrt();
			const Eigen::MatrixXd Ainv = es.operatorInverseSqrt();
			const Eigen::MatrixXd RZ = this->Retract(Z);
			const Eigen::MatrixXd RZtY = RZ.transpose() * Y;
			const Eigen::MatrixXd Q = RZtY - RZtY.transpose();
			const Eigen::MatrixXd Lambda = StiefelTemplateSylvester(A, Q);
			return RZ * Lambda + ( Eigen::MatrixXd::Identity(nrows, nrows) - RZ * RZ.transpose() ) * Y * Ainv;
		}
	}else if ( this->Geodesic == "QR" ){
		if constexpr (!IsGeneralized){
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
	}
	__Check_Geodesic_Func__
	return Y;
}

template<bool IsGeneralized>
Eigen::MatrixXd StiefelTemplate<IsGeneralized>::TransportManifold(Eigen::MatrixXd X, Manifold& N) const{
	__Check_Vec_Transport__
	const Eigen::MatrixXd Z = this->InverseRetract(N);
	return this->TransportTangent(X, Z);
}

template<bool IsGeneralized>
Eigen::MatrixXd StiefelTemplate<IsGeneralized>::symf(Eigen::MatrixXd PtBX) const{
	return 0.5 * ( PtBX + PtBX.transpose() );
}

template<bool IsGeneralized>
Eigen::MatrixXd StiefelTemplate<IsGeneralized>::TangentProjection(Eigen::MatrixXd X) const{
	if constexpr (IsGeneralized) return X - this->P * symf(this->P.transpose() * this->B * X);
	return X - this->P * symf(this->P.transpose() * X);
}

template<bool IsGeneralized>
Eigen::MatrixXd StiefelTemplate<IsGeneralized>::TangentPurification(Eigen::MatrixXd X) const{
	return this->TangentProjection(X);
}

template<bool IsGeneralized>
void StiefelTemplate<IsGeneralized>::setPoint(Eigen::MatrixXd p, bool purify){
	if ( purify ){
		if constexpr (IsGeneralized){
			Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(p.transpose() * this->B * p);
			p = ( p * es.operatorInverseSqrt() ).eval();
		}else{
			Eigen::BDCSVD<Eigen::MatrixXd, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(p);
			p = svd.matrixU() * svd.matrixV().transpose();
		}
	}
	this->P = p;
}

template<bool IsGeneralized>
void StiefelTemplate<IsGeneralized>::getGradient(){
	if constexpr (IsGeneralized) this->Gr = this->TangentProjection(this->Sharp(this->Ge));
	else this->Gr = this->TangentProjection(this->Ge);
}

template<bool IsGeneralized>
Eigen::MatrixXd StiefelTemplate<IsGeneralized>::getHessian(Eigen::MatrixXd HeX, Eigen::MatrixXd X) const{
	if constexpr (IsGeneralized){
		return TangentProjection(
			this->Binv * HeX - X * this->P.transpose() * (
				this->Ge - this->B * this->Gr
			)
		);
	}else{
		return this->TangentProjection(HeX - X * symf(this->P.transpose() * this->Ge));
	}
}

template<bool IsGeneralized>
std::unique_ptr<Manifold> StiefelTemplate<IsGeneralized>::Clone() const{
	return std::make_unique<StiefelTemplate<IsGeneralized>>(*this);
}

template<bool IsGeneralized>
std::shared_ptr<Manifold> StiefelTemplate<IsGeneralized>::Share() const{
	return std::make_shared<StiefelTemplate<IsGeneralized>>(*this);
}

StiefelFromTemplate::StiefelFromTemplate(Eigen::MatrixXd p, std::string geodesic):
	StiefelTemplate<false>(std::move(p), std::move(geodesic)){}

std::unique_ptr<Manifold> StiefelFromTemplate::Clone() const{
	return std::make_unique<StiefelFromTemplate>(*this);
}

std::shared_ptr<Manifold> StiefelFromTemplate::Share() const{
	return std::make_shared<StiefelFromTemplate>(*this);
}

GeneralizedStiefelFromTemplate::GeneralizedStiefelFromTemplate(Eigen::MatrixXd p, Eigen::MatrixXd B, std::string geodesic):
	StiefelTemplate<true>(std::move(p), std::move(B), std::move(geodesic)){}

std::unique_ptr<Manifold> GeneralizedStiefelFromTemplate::Clone() const{
	return std::make_unique<GeneralizedStiefelFromTemplate>(*this);
}

std::shared_ptr<Manifold> GeneralizedStiefelFromTemplate::Share() const{
	return std::make_shared<GeneralizedStiefelFromTemplate>(*this);
}

template class StiefelTemplate<false>;
template class StiefelTemplate<true>;

}