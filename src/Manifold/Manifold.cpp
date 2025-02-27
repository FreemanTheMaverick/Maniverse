#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#endif
#include <Eigen/Dense>
#include <typeinfo>
#include <memory>

#include "../Macro.h"

#include "Manifold.h"

#include <iostream>


Manifold::Manifold(EigenMatrix p, bool matrix_free){
	this->P.resize(p.rows(), p.cols());
	this->Ge.resize(p.rows(), p.cols());
	this->Gr.resize(p.rows(), p.cols());
	this->P = p;
	this->MatrixFree = matrix_free;
	if (this->MatrixFree){
		this->Hem.resize(0, 0);
		this->Hrm.resize(0);
	}
}

int Manifold::getDimension() const{
	__Not_Implemented__
	return 0;
}

double Manifold::Inner(EigenMatrix X, EigenMatrix Y) const{
	__Not_Implemented__
	return X.rows() * Y.cols() * 0; // Avoiding the unused-variable warning
}

static std::tuple<EigenVector, EigenMatrix> ThinEigen(EigenMatrix A, int m){
	// n - Total number of eigenvalues
	// m - Number of non-trivial eigenvalues
	const int n = A.rows();
	Eigen::SelfAdjointEigenSolver<EigenMatrix> es;
	es.compute(A);
	std::vector<std::tuple<double, EigenVector>> eigen_tuples;
	eigen_tuples.reserve(n);
	for ( int i = 0; i < n; i++ )
		eigen_tuples.push_back(std::make_tuple(es.eigenvalues()(i), es.eigenvectors().col(i)));
	std::sort( // Sorting the eigenvalues in decreasing order of magnitude.
			eigen_tuples.begin(), eigen_tuples.end(),
			[](std::tuple<double, EigenVector>& a, std::tuple<double, EigenVector>& b){
				return std::abs(std::get<0>(a)) > std::abs(std::get<0>(b));
			}
	); // Now the eigenvalues closest to zero are in the back.
	eigen_tuples.resize(m); // Deleting them.
	std::sort( // Resorting the eigenvalues in increasing order.
			eigen_tuples.begin(), eigen_tuples.end(),
			[](std::tuple<double, EigenVector>& a, std::tuple<double, EigenVector>& b){
				return std::get<0>(a) < std::get<0>(b);
			}
	);
	EigenVector eigenvalues = EigenZero(m, 1);
	EigenMatrix eigenvectors = EigenZero(n, m);
	for ( int i = 0; i < m; i++ ){
		eigenvalues(i) = std::get<0>(eigen_tuples[i]);
		eigenvectors.col(i) = std::get<1>(eigen_tuples[i]);
	}
	return std::make_tuple(eigenvalues, eigenvectors);
}

void Manifold::getBasisSet(){
	const int nrows = this->P.rows();
	const int ncols = this->P.cols();
	const int size = nrows * ncols;
	const int rank = this->getDimension();
	EigenMatrix euclidean_basis = EigenZero(nrows, ncols);
	std::vector<EigenMatrix> unorthogonal_basis_set(size, EigenZero(nrows, ncols));
	for ( int i = 0, n = 0; i < nrows; i++ ) for ( int j = 0; j < ncols; j++ , n++){
		euclidean_basis(i, j) = 1;
		unorthogonal_basis_set[n] = TangentProjection(euclidean_basis);
		euclidean_basis(i, j) = 0;
	}
	EigenMatrix gram = EigenZero(size, size);
	for ( int i = 0; i < size; i++ ) for ( int j = 0; j <= i; j++ ){
		gram(i, j) = gram(j, i) = this->Inner(unorthogonal_basis_set[i], unorthogonal_basis_set[j]);
	}
	auto [Sigma, U] = ThinEigen(gram, rank);
	const EigenMatrix C = U * Sigma.cwiseSqrt().asDiagonal();
	this->BasisSet.resize(rank);
	for ( int i = 0; i < rank; i++ ){
		this->BasisSet[i].resize(nrows, ncols); this->BasisSet[i].setZero();
		for ( int j = 0; j < size; j++ ){
			this->BasisSet[i] += C(j, i) * unorthogonal_basis_set[j].reshaped<Eigen::RowMajor>(nrows, ncols);
		}
	}
}

std::vector<std::tuple<double, EigenMatrix>> Diagonalize(
		EigenMatrix& A, std::vector<EigenMatrix>& basis_set){
	Eigen::SelfAdjointEigenSolver<EigenMatrix> es;
	es.compute( ( A + A.transpose() ) / 2 );
	const EigenMatrix Lambda = es.eigenvalues();
	const EigenMatrix Y = es.eigenvectors();
	const int nrows = basis_set[0].rows();
	const int ncols = basis_set[0].cols();
	const int rank = basis_set.size();
	std::vector<std::tuple<double, EigenMatrix>> hrm(rank, std::tuple(0, EigenZero(nrows, ncols)));
	for ( int i = 0; i < rank; i++ ){
		std::get<0>(hrm[i]) = Lambda(i);
		for ( int j = 0; j < rank; j++ ){
			std::get<1>(hrm[i]) += basis_set[j] * Y(j, i);
		}
	}
	return hrm;
}

void Manifold::getHessianMatrix(){
	// Representing the Riemannian hessian with the orthogonal basis set
	const int rank = this->getDimension();
	EigenMatrix hrm = EigenZero(rank, rank);
	for ( int i = 0; i < rank; i++ ) for ( int j = 0; j <= i; j++ ){
		hrm(i, j) = hrm(j, i) = this->Inner(this->BasisSet[i], this->Hr(this->BasisSet[j]));
	}

	// Diagonalizing the Riemannian hessian and representing the eigenvectors in Euclidean space
	this->Hrm = Diagonalize(hrm, this->BasisSet);

	// Updating the Riemannian hessian operator
	this->Hr = [&hrm = this->Hrm](EigenMatrix v){ // Passing reference instead of value to std::function, so that the eigenvalues can be modified elsewhere without rewriting this part.
		EigenMatrix Hv = EigenZero(v.rows(), v.cols());
		for ( auto [eigenvalue, eigenvector] : hrm ){
			Hv += eigenvalue * eigenvector.cwiseProduct(v).sum() * eigenvector;
		}
		return Hv;
	};
}

EigenMatrix Manifold::Exponential(EigenMatrix X) const{
	__Not_Implemented__
	return EigenZero(X.rows(), X.cols());
}

EigenMatrix Manifold::Logarithm(Manifold& N) const{
	__Not_Implemented__
	return EigenZero(N.P.rows(), N.P.cols());
}

EigenMatrix Manifold::TangentProjection(EigenMatrix A) const{
	__Not_Implemented__
	return EigenZero(A.rows(), A.cols());
}

EigenMatrix Manifold::TangentPurification(EigenMatrix A) const{
	__Not_Implemented__
	return EigenZero(A.rows(), A.cols());
}

EigenMatrix Manifold::TransportTangent(EigenMatrix X, EigenMatrix Y) const{
	__Not_Implemented__
	return EigenZero(X.rows(), Y.cols());
}

EigenMatrix Manifold::TransportManifold(EigenMatrix X, Manifold& N) const{
	__Not_Implemented__
	return EigenZero(X.rows(), N.P.cols());
}

void Manifold::Update(EigenMatrix p, bool purify){
	if ( purify ? p.rows() : p.cols() ){ // To avoid the unused-variable warning.
		__Not_Implemented__
	}else{
		__Not_Implemented__
	}
}

void Manifold::getGradient(){
	__Not_Implemented__
}

void Manifold::getHessian(){
	__Not_Implemented__
}

std::unique_ptr<Manifold> Manifold::Clone() const{
	__Not_Implemented__
	return std::make_unique<Manifold>(*this);
}

#ifdef __PYTHON__
void Init_Manifold(pybind11::module_& m){
	pybind11::class_<Manifold>(m, "Manifold")
		.def_readwrite("Name", &Manifold::Name)
		.def_readwrite("P", &Manifold::P)
		.def_readwrite("Ge", &Manifold::Ge)
		.def_readwrite("Gr", &Manifold::Gr)
		.def_readwrite("MatrixFree", &Manifold::MatrixFree)
		.def_readwrite("Hem", &Manifold::Hem)
		.def_readwrite("Hrm", &Manifold::Hrm)
		.def_readwrite("He", &Manifold::He)
		.def_readwrite("Hr", &Manifold::Hr)
		.def_readwrite("BasisSet", &Manifold::BasisSet)
		.def(pybind11::init<EigenMatrix, bool>())
		.def("getDimension", &Manifold::getDimension)
		.def("Inner", &Manifold::Inner)
		.def("getBasisSet", &Manifold::getBasisSet)
		.def("getHessianMatrix", &Manifold::getHessianMatrix)
		.def("Exponential", &Manifold::Exponential)
		.def("Logarithm", &Manifold::Logarithm)
		.def("TangentProjection", &Manifold::TangentProjection)
		.def("TangentPurification", &Manifold::TangentPurification)
		.def("TransportTangent", &Manifold::TransportTangent)
		.def("TransportManifold", &Manifold::TransportManifold)
		.def("Update", &Manifold::Update)
		.def("getGradient", &Manifold::getGradient)
		.def("getHessian", &Manifold::getHessian)
		.def("Clone", &Manifold::Clone);
	m.def("Diagonalize", &Diagonalize);
}
#endif
