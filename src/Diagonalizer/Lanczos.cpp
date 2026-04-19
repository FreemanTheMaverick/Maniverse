#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#endif

#include <Eigen/Dense>
#include <tuple>
#include <vector>
#include <random>
#include <cstdio>
#include <chrono>
#include<iostream>

#include "../Macro.h"
#include "../Manifold/Manifold.h"

namespace Maniverse{

std::tuple<std::vector<double>, std::vector<Eigen::VectorXd>> Lanczos(Iterate& M, int m, int output){
	if (output){
		std::printf("********************** Lanczos diagonalization of hessian **********************\n\n");
		std::printf("Manifold: %s\n", M.getName().c_str());
		std::printf("Dimension number: %d\n", M.getDimension());
		std::printf("Number of eigenpairs attempted: %d\n", m);
	}
	const auto all_start = __now__;
	std::mt19937 gen(114514);
	std::uniform_real_distribution<double> dis(-3, 3);
	Eigen::MatrixXd V(M.TotalSize, m);
	for ( int i = 0; i < M.TotalSize; i++ ) V(i, 0) = dis(gen);
	V.col(0) = M.ConstraintProjection(M.TangentProjection(V.col(0)));
	V.col(0) /= std::sqrt(M.Inner(V.col(0), V.col(0)));
	Eigen::MatrixXd T = Eigen::MatrixXd::Zero(m, m);
	Eigen::VectorXd w(M.TotalSize);
	for ( int j = 0; j < m; j++ ){
		if (output) std::printf("\nIteration %d:\n", j);
		const auto iter_start = __now__;
		double alpha = 0; double beta = 0;
		if ( j > 0 ){
			for ( int k = 0; k < j; k++ ) w -= V.col(k) * M.Inner(w, V.col(k));
			w = M.ConstraintProjection(M.TangentProjection(w));
			beta = T(j, j - 1) = T(j - 1, j) = std::sqrt(M.Inner(w, w));
			if (output) std::printf("Beta = %f\n", beta);
			if ( beta < 1e-10 ){
				if (output) std::printf("Early stop due to the small Beta. This may indicates degeneracy in the eigenpairs.\n");
				m = j;
				goto ShowTime;
			}
			V.col(j) = w / beta;
			w = - beta * V.col(j - 1);
		}
		w += M.ConstraintProjectedHessian(V.col(j));
		alpha = T(j, j) = M.Inner( w, V.col(j) );
		if (output) std::printf("Alpha = %f\n", alpha);
		w -= alpha * V.col(j);
		ShowTime:
		if (output) std::printf("Elapsed time: %f seconds for current iteration; %f seconds in total\n", __duration__(iter_start, __now__), __duration__(all_start, __now__));
	}
	T = T.topLeftCorner(m, m).eval();
	V = V.leftCols(m).eval();
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(T);
	Eigen::VectorXd Evals = es.eigenvalues();
	if (output){
		std::printf("\n%d Eigenvalues found:", m);
		for ( int j = 0; j < m; j++ ) std::printf(" %f", Evals[j]);
		std::printf("\n");
	}
	Eigen::MatrixXd Evecs = V * es.eigenvectors();
	std::vector<Eigen::VectorXd> Evecs_vec(m);
	for ( int j = 0; j < m; j++ ) Evecs_vec[j] = Evecs.col(j);
	return std::make_tuple(std::vector<double>(Evals.data(), Evals.data() + m), Evecs_vec);
}

#ifdef __PYTHON__
void Init_Lanczos(pybind11::module_& m){
	m.def("Lanczos", &Lanczos);
}
#endif

}
