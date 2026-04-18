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

#include "../Macro.h"
#include "../Manifold/Manifold.h"

namespace Maniverse{

std::tuple<std::vector<double>, std::vector<Eigen::VectorXd>> Lanczos(Iterate& M, int m, double beta_min, int output){
	if (output){
		std::printf("********************** Lanczos diagonalization of hessian **********************\n\n");
		std::printf("Manifold: %s\n", M.getName().c_str());
		std::printf("Dimension number: %d\n", M.getDimension());
		std::printf("Number of eigenpairs attempted: %d\n", m);
		std::printf("Minimal acceptable beta: %E\n", beta_min);
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
		double beta = 0;
		if ( j > 0 ){
			OrthogonalizeW:
			for ( int k = 0; k < j; k++ ) w -= V.col(k) * M.Inner(w, V.col(k));
			beta = T(j, j - 1) = T(j - 1, j) = std::sqrt(M.Inner(w, w));
			if (output) std::printf("Beta = %f\n", beta);
			if ( beta < beta_min ){
				if (output) std::printf("Beta is too small. Switching to a random vector.\n");
				for ( int i = 0; i < M.TotalSize; i++ ) w(i) = dis(gen);
				w = M.ConstraintProjection(M.TangentProjection(w));
				goto OrthogonalizeW;
			}
			V.col(j) = w / beta;
		}
		w = M.ConstraintProjectedHessian(V.col(j));
		if ( j > 0 ) w -= beta * V.col(j - 1);
		const double alpha = T(j, j) = M.Inner( w, V.col(j) );
		if (output) std::printf("Alpha = %f\n", alpha);
		w -= alpha * V.col(j);
		if (output) std::printf("Elapsed time: %f seconds for current iteration; %f seconds in total\n\n", __duration__(iter_start, __now__), __duration__(all_start, __now__));
	}
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(T);
	Eigen::VectorXd Evals = es.eigenvalues();
	if (output){
		std::printf("\nEigenvalues:");
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
