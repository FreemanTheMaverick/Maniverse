#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <cstdio>
#include <chrono>
#include <cassert>
#include <string>
#include <tuple>
#include <algorithm>

#include "../Macro.h"
#include "../Manifold/Manifold.h"

#include <iostream>


std::tuple<EigenVector, EigenMatrix> ThinEigen(EigenMatrix A, int m){
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

std::tuple<EigenVector, EigenMatrix> ExactDiagonalization(Manifold& M){
	const int rank = M.getDimension();
	const EigenMatrix gram = M.getGram();
	const auto [Sigma, U] = ThinEigen(gram, rank);
	const EigenMatrix C = U * Sigma.cwiseSqrt().asDiagonal();
	const EigenMatrix B = C.transpose() * M.Hrm * C;
	const auto [Lambda, Y] = ThinEigen(B, rank);
	const EigenMatrix X = C * Y;
	return std::make_tuple(Lambda, X);
}

void Init_EigenSolver(pybind11::module_& m){
	m.def("ExactDiagonalization", &ExactDiagonalization);
}
