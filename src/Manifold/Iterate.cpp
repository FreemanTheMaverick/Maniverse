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

Iterate::Iterate(std::vector<std::shared_ptr<Manifold>> Ms, bool matrix_free){
	const int nMs = (int)Ms.size();
	this->Ms.clear();
	for ( int iM = 0; iM < nMs; iM++ ) this->Ms.push_back(Ms[iM]->Clone());

	int nrows = 0;
	int ncols = 0;
	for ( int iM = 0; iM < nMs; iM++ ){
		this->BlockParameters.push_back(std::make_tuple(
				nrows, ncols,
				this->Ms[iM]->P.rows(),
				this->Ms[iM]->P.cols()
		));
		nrows += this->Ms[iM]->P.rows();
		ncols += this->Ms[iM]->P.cols();
	}

	this->Point.resize(nrows, ncols); this->Point.setZero();
	this->Gradient.resize(nrows, ncols); this->Gradient.setZero();
	for ( int iM = 0; iM < (int)Ms.size(); iM++ ){
		GetBlock(this->Point, iM) = Ms[iM]->P;
		GetBlock(this->Gradient, iM) = Ms[iM]->Gr;
	}

	this->MatrixFree = matrix_free;
}

Iterate::Iterate(const Iterate& another_iterate){
	for ( auto& M : another_iterate.Ms ) this->Ms.push_back(M->Clone());
	this->Point = another_iterate.Point;
	this->Gradient = another_iterate.Gradient;
	this->Hessian = another_iterate.Hessian;
	this->MatrixFree = another_iterate.MatrixFree;
	this->BasisSet = another_iterate.BasisSet;
	this->HessianMatrix = another_iterate.HessianMatrix;
	this->BlockParameters = another_iterate.BlockParameters;
}

std::string Iterate::getName() const{
	std::string name = "";
	for ( int iM = 0; iM < (int)this->Ms.size(); iM++ ){
		if ( iM > 0 ) name += " * ";
		name += Ms[iM]->Name;
	}
	return name;
}

int Iterate::getDimension() const{
	int ndims = 0;
	for ( int iM = 0; iM < (int)this->Ms.size(); iM++ )
		 ndims += Ms[iM]->getDimension();
	return ndims;
}

double Iterate::Inner(EigenMatrix X, EigenMatrix Y) const{
	double inner = 0;
	for ( int iM = 0; iM < (int)this->Ms.size(); iM++ ){
		inner += this->Ms[iM]->Inner(GetBlock(X, iM), GetBlock(Y, iM));
	}
	return inner;
}

EigenMatrix Iterate::Exponential(EigenMatrix X) const{
	EigenMatrix Exp = EigenZero(X.rows(), X.cols());
	for ( int iM = 0; iM < (int)this->Ms.size(); iM++ ){
		GetBlock(Exp, iM) = this->Ms[iM]->Exponential(GetBlock(X, iM));
	}
	return Exp;
}

EigenMatrix Iterate::TangentProjection(EigenMatrix A) const{
	EigenMatrix X = EigenZero(A.rows(), A.cols());
	for ( int iM = 0; iM < (int)this->Ms.size(); iM++ ){
		GetBlock(X, iM) = this->Ms[iM]->TangentProjection(GetBlock(A, iM));
	}
	return X;
}

EigenMatrix Iterate::TangentPurification(EigenMatrix A) const{
	EigenMatrix X = EigenZero(A.rows(), A.cols());
	for ( int iM = 0; iM < (int)this->Ms.size(); iM++ ){
		GetBlock(X, iM) = this->Ms[iM]->TangentPurification(GetBlock(A, iM));
	}
	return X;
}

EigenMatrix Iterate::TransportManifold(EigenMatrix A, Iterate& N) const{
	EigenMatrix B = EigenZero(A.rows(), A.cols());
	for ( int iM = 0; iM < (int)this->Ms.size(); iM++ ){
		GetBlock(B, iM) = this->Ms[iM]->TransportManifold(GetBlock(A, iM), *(N.Ms[iM]));
	}
	return B;
}

void Iterate::setPoint(std::vector<EigenMatrix> ps, bool purify){
	if ( ps.size() != this->Ms.size() ) throw std::runtime_error("Wrong number of Points!");
	for ( int iM = 0; iM < (int)this->Ms.size(); iM++ ){
		this->Ms[iM]->setPoint(ps[iM], purify);
		GetBlock(this->Point, iM) = this->Ms[iM]->P;
	}
}

void Iterate::setGradient(std::vector<EigenMatrix> gs){
	if ( gs.size() != this->Ms.size() ) throw std::runtime_error("Wrong number of gradients!");
	for ( int iM = 0; iM < (int)this->Ms.size(); iM++ ){
		this->Ms[iM]->Ge = gs[iM];
		this->Ms[iM]->getGradient();
		GetBlock(this->Gradient, iM) = this->Ms[iM]->Gr;
	}
}

std::vector<EigenMatrix> Iterate::getPoint() const{
	std::vector<EigenMatrix> ps;
	for ( int iM = 0; iM < (int)this->Ms.size(); iM++ ){
		ps.push_back(GetBlock(this->Point, iM));
	}
	return ps;
}

std::vector<EigenMatrix> Iterate::getGradient() const{
	std::vector<EigenMatrix> gs;
	for ( int iM = 0; iM < (int)this->Ms.size(); iM++ ){
		gs.push_back(GetBlock(this->Gradient, iM));
	}
	return gs;
}

void Iterate::setHessian(std::vector<std::function<EigenMatrix (EigenMatrix)>> hs){
	const int nMs = (int)this->Ms.size();
	if ( (int)hs.size() != nMs * nMs ) throw std::runtime_error("Wrong number of hessians!");
	for ( int iM = 0, khess = 0; iM < nMs; iM++ ) for ( int jM = 0; jM < nMs; jM++, khess++ ){
		hs[khess] = this->Ms[iM]->getHessian(hs[khess], iM == jM);
	}
	this->Hessian = [nMs, hs, BlockParameters = this->BlockParameters](EigenMatrix X){
		EigenMatrix HX = EigenZero(X.rows(), X.cols());
		for ( int iM = 0, khess = 0; iM < nMs; iM++ ) for ( int jM = 0; jM < nMs; jM++, khess++ ){
			GetBlock(HX, iM) += hs[khess](GetBlock(X, jM));
		}
		return HX;
	};
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

void Iterate::getBasisSet(){
	const int nrows = this->Point.rows();
	const int ncols = this->Point.cols();
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

void Iterate::getHessianMatrix(){
	// Representing the Riemannian hessian with the orthogonal basis set
	const int rank = this->getDimension();
	EigenMatrix hrm = EigenZero(rank, rank);
	for ( int i = 0; i < rank; i++ ) for ( int j = 0; j <= i; j++ ){
		hrm(i, j) = hrm(j, i) = this->Inner(this->BasisSet[i], this->Hessian(this->BasisSet[j]));
	}

	// Diagonalizing the Riemannian hessian and representing the eigenvectors in Euclidean space
	this->HessianMatrix = Diagonalize(hrm, this->BasisSet);

	// Updating the Riemannian hessian operator
	this->Hessian = [&hrm = this->HessianMatrix](EigenMatrix v){ // Passing reference instead of value to std::function, so that the eigenvalues can be modified elsewhere without rewriting this part.
		EigenMatrix Hv = EigenZero(v.rows(), v.cols());
		for ( auto [eigenvalue, eigenvector] : hrm ){
			Hv += eigenvalue * eigenvector.cwiseProduct(v).sum() * eigenvector;
		}
		return Hv;
	};
}

#ifdef __PYTHON__
void Init_Iterate(pybind11::module_& m){
	pybind11::class_<Iterate>(m, "Iterate")
		.def_readonly("Ms", &Iterate::Ms)
		.def_readwrite("Point", &Iterate::Point)
		.def_readwrite("Gradient", &Iterate::Gradient)
		.def_readwrite("Hessian", &Iterate::Hessian)
		.def_readwrite("MatrixFree", &Iterate::MatrixFree)
		.def_readwrite("BasisSet", &Iterate::BasisSet)
		.def_readwrite("HessianMatrix", &Iterate::HessianMatrix)
		.def_readwrite("BlockParameters", &Iterate::BlockParameters)
		.def(pybind11::init<std::vector<std::shared_ptr<Manifold>>, bool>())
		.def(pybind11::init<const Iterate&>())
		.def("getName", &Iterate::getName)
		.def("getDimension", &Iterate::getDimension)
		.def("Inner", &Iterate::Inner)
		.def("Exponential", &Iterate::Exponential)
		.def("TangentProjection", &Iterate::TangentProjection)
		.def("TangentPurification", &Iterate::TangentPurification)
		.def("TransportManifold", &Iterate::TransportManifold)
		.def("setPoint", &Iterate::setPoint)
		.def("setGradient", &Iterate::setGradient)
		.def("setHessian", &Iterate::setHessian)
		.def("getBasisSet", &Iterate::getBasisSet)
		.def("getHessianMatrix", &Iterate::getHessianMatrix);
	m.def("Diagonalize", &Diagonalize);
}
#endif
