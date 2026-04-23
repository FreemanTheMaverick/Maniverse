#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#endif

#include <Eigen/Dense>
#include <tuple>
#include <vector>
#include <random>
#include <cstdio>
#include <chrono>
#include <functional>

#include "../Macro.h"
#include "../Manifold/Manifold.h"

namespace Maniverse{

// https://doi.org/10.1137/19M1279691
std::tuple<std::vector<double>, std::vector<Eigen::VectorXd>> Lanczos(
		std::function<double (Eigen::VectorXd, Eigen::VectorXd)> dot,
		std::function<Eigen::VectorXd (Eigen::VectorXd)> proj,
		std::function<Eigen::VectorXd (Eigen::VectorXd)> A,
		Eigen::VectorXd b,
		std::function<Eigen::VectorXd (Eigen::VectorXd)> P,
		int m, bool output){
	const int totalsize = (int)b.size();
	if (output){
		std::printf("*************************** Lanczos diagonalization ***************************\n\n");
		std::printf("Size of each vector: %d\n", totalsize);
		std::printf("Number of eigenpairs attempted: %d\n", m);
	}
	const auto all_start = __now__;
	Eigen::MatrixXd V(totalsize, m);
	Eigen::MatrixXd T = Eigen::MatrixXd::Zero(m, m);
	Eigen::VectorXd t = b;
	Eigen::VectorXd w_last = Eigen::VectorXd::Zero(totalsize);
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
	Eigen::VectorXd Evals = Eigen::VectorXd::Zero(totalsize);
	bool stop = 0;
	for ( int j = 0; j < m && !stop; j++ ){
		if (output) std::printf("\nIteration %d:\n", j);
		const auto iter_start = __now__;
		for ( int k = 0; k < j; k++ ) t -= V.col(k) * dot(t, V.col(k));
		t = proj(t);
		const Eigen::VectorXd y = P(t);
		const double beta = std::sqrt(dot(y, t));
		if (output) std::printf("Beta = %f\n", beta);
		if ( beta < 1e-10 ){
			if (output) std::printf("Early stop due to the small Beta. This may indicate degeneracy in the eigenpairs.\n");
			stop = 1;
		}else{
			if ( j > 0 ) T(j, j - 1) = T(j - 1, j) = beta;
			const Eigen::VectorXd w = t / beta;
			V.col(j) = y / beta;
			const Eigen::VectorXd Av = A(V.col(j));
			const double alpha = T(j, j) = dot( Av, V.col(j) );
			t = Av - alpha * w - beta * w_last;
			w_last = w;
			es.compute(T.topLeftCorner(j + 1, j + 1));
			Evals = es.eigenvalues();
			if (output){
				std::printf("%d Eigenvalues found:", j + 1);
				for ( int i = 0; i < j + 1; i++ ) std::printf(" %f", Evals(i));
				std::printf("\n");
			}
		}
		if (output) std::printf("Elapsed time: %f seconds for current iteration; %f seconds in total\n", __duration__(iter_start, __now__), __duration__(all_start, __now__));
	}
	m = (int)Evals.size();
	V = V.leftCols(m).eval();
	const Eigen::MatrixXd Evecs = V * es.eigenvectors();
	std::vector<Eigen::VectorXd> Evecs_vec(m);
	for ( int j = 0; j < m; j++ ) Evecs_vec[j] = Evecs.col(j);
	return std::make_tuple(std::vector<double>(Evals.data(), Evals.data() + m), Evecs_vec);
}

std::tuple<std::vector<double>, std::vector<Eigen::VectorXd>> Lanczos(Iterate& M, int m, bool constraint, bool output){
	if (output){
		std::printf("Configuring Lanczos diagonalization of hessian\n");
		std::printf("Manifold: %s\n", M.getName().c_str());
		std::printf("Dimension number: %d\n", M.getDimension());
		if (constraint) std::printf("Extra constraint: Yes\n");
		else std::printf("Extra constraint: No\n");
	}

	// Inner product
	const auto dot = [&M](Eigen::VectorXd X, Eigen::VectorXd Y) -> double{
		return M.Inner(X, Y);
	};

	// Projector
	const auto proj = constraint ?
		std::function<Eigen::VectorXd (Eigen::VectorXd)>([&M](Eigen::VectorXd X) -> Eigen::VectorXd{ return M.ConstraintProjection(M.TangentProjection(X)); }) :
		std::function<Eigen::VectorXd (Eigen::VectorXd)>([&M](Eigen::VectorXd X) -> Eigen::VectorXd{ return M.TangentProjection(X); }) ;

	// Hessian
	const auto A = constraint ?
		std::function<Eigen::VectorXd (Eigen::VectorXd)>([&M](Eigen::VectorXd X) -> Eigen::VectorXd{ return M.ConstraintProjectedHessian(X); }) :
		std::function<Eigen::VectorXd (Eigen::VectorXd)>([&M](Eigen::VectorXd X) -> Eigen::VectorXd{ return M.Hessian(X); }) ;

	// Trial vector
	std::mt19937 gen(114514);
	std::uniform_real_distribution<double> dis(-3, 3);
	Eigen::VectorXd b(M.TotalSize);
	for ( int i = 0; i < M.TotalSize; i++ ) b(i) = dis(gen);
	b = M.ConstraintProjection(M.TangentProjection(b));
	b /= std::sqrt(M.Inner(b, b));

	// Preconditioner
	const auto P = constraint ?
		std::function<Eigen::VectorXd (Eigen::VectorXd)>([&M](Eigen::VectorXd X) -> Eigen::VectorXd{ return M.ConstraintProjectedPreconditioner(X); }) :
		std::function<Eigen::VectorXd (Eigen::VectorXd)>([&M](Eigen::VectorXd X) -> Eigen::VectorXd{ return M.Preconditioner(X); }) ;

	return Lanczos(dot, proj, A, b, P, m, output);
}

#ifdef __PYTHON__
void Init_Lanczos(pybind11::module_& m){
	m.def("Lanczos", (std::tuple<std::vector<double>, std::vector<Eigen::VectorXd>>(*)(std::function<double(Eigen::VectorXd,Eigen::VectorXd)>, std::function<Eigen::VectorXd(Eigen::VectorXd)>, std::function<Eigen::VectorXd(Eigen::VectorXd)>, Eigen::VectorXd, std::function<Eigen::VectorXd(Eigen::VectorXd)>, int, bool)) &Lanczos);
	m.def("Lanczos", (std::tuple<std::vector<double>, std::vector<Eigen::VectorXd>>(*)(Iterate&, int, bool, bool)) &Lanczos);
}
#endif

}
