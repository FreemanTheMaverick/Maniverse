#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#endif

#include <Eigen/Dense>
#include <cmath>
#include <tuple>
#include <vector>
#include <deque>
#include <cstdio>
#include <chrono>

#include "../Macro.h"
#include "../Manifold/Manifold.h"

#include "Anderson.h"

namespace Maniverse{

// https://doi.org/10.1287/moor.2023.0284
bool Anderson(
		Iterate& M,
		std::tuple<double, double, double> tol,
		double beta, int max_mem, int max_iter,
		int output){

	auto [tol0, tol1, tol2] = tol;
	if (output > 0){
		std::printf("******************************* Anderson Mixing *******************************\n\n");
		std::printf("Manifold: %s\n", M.getName().c_str());
		std::printf("Dimension number: %d\n", M.getDimension());
		std::printf("Maximum number of iterations: %d\n", max_iter);
		std::printf("Maximum memory of previous iterations: %d\n", max_mem);
		std::printf("Convergence threshold:\n");
		std::printf("| Target change (T. C.)               : %E\n", tol0);
		std::printf("| Residual norm (Resi.)               : %E\n", tol1);
		std::printf("| Independent variable update (V. U.) : %E\n\n", tol2);
	}

	const auto all_start = __now__;

	double oldL = 0;
	double actual_delta_L = 0;

	std::vector<Eigen::MatrixXd> P = M.getPoint();
	std::vector<Eigen::MatrixXd> R = M.getPoint();
	Eigen::VectorXd Pvec = M.Point;
	Eigen::VectorXd S = Eigen::VectorXd::Zero(Pvec.size());
	Eigen::VectorXd Rvec = Eigen::VectorXd::Zero(Pvec.size());

	std::deque<Eigen::VectorXd> Ss;
	std::deque<Eigen::VectorXd> Ys;

	bool converged = 0;

	for ( int iiter = 0; ( iiter < max_iter ) && ( ! converged ); iiter++ ){
		if (output) std::printf("Iteration %d\n", iiter);
		const auto iter_start = __now__;

		Eigen::VectorXd oldRvec = M.TransportTangent(Rvec, S);
		M.Calculate(P, {0, 1});
		actual_delta_L = M.Value - oldL;
		oldL = M.Value;
		if (output) std::printf("Target = %.10f\n", M.Value);

		// Transporting previous vectors I
		if ( (int)Ss.size() == max_mem ){
			Ss.pop_front();
			Ys.pop_front();
		}
		for ( int i = 0 ; i < (int)Ss.size(); i++ ){
			Ss[i] = M.TransportTangent(Ss[i], S);
			Ys[i] = M.TransportTangent(Ys[i], S);
		}
		if ( iiter > 0 ) Ss.push_back(M.TransportTangent(S, S));
		const int size = (int)Ss.size();
		if (output) std::printf("Current memory size: %d\n", size);

		// Checking convergence
		M.setPoint(P, 1);
		M.setGradient();
		R = M.getGradient();
		Rvec = M.Gradient;

		const double Rnorm = std::sqrt( M.Inner(Rvec, Rvec) );
		const double Snorm = std::sqrt( M.Inner(S, S) );
		if ( Rnorm < tol1 ){
			if ( iiter == 0 ) converged = 1;
			else if ( std::abs(actual_delta_L) < tol0 && Snorm < tol2 ) converged = 1;
		}
		if (output){
			std::printf("Convergence info: current / threshold / converged?\n");
			std::printf("| Target    change: % E / %E / %s\n", actual_delta_L, tol0, __True_False__(std::abs(actual_delta_L) < tol0));
			std::printf("| Residual    norm: % E / %E / %s\n", Rnorm, tol1, __True_False__(Rnorm < tol1));
			std::printf("| Step length norm: % E / %E / %s\n", Snorm, tol2, __True_False__(Snorm < tol2));
			if ( converged ) std::printf("| Converged!\n");
			else std::printf("| Not converged yet!\n");
		}

		// Transporting previous vectors II
		if ( iiter > 0 ) Ys.push_back(Rvec - oldRvec);

		if ( size > 0 ){
			// Solving for the extrapolation vector
			// https://dx.doi.org/10.13471/j.cnki.acta.snus.2023A035
			Eigen::MatrixXd YtY = Eigen::MatrixXd::Zero(size, size);
			for ( int i = 0; i < size; i++ ) for ( int j = i; j < size; j++ ){
				YtY(i, j) = YtY(j, i) = Ys[i].cwiseProduct(Ys[j]).sum();
			}
			const Eigen::MatrixXd YtYinv = YtY.ldlt().solve(Eigen::MatrixXd::Identity(size, size));
			Eigen::VectorXd YtR = Eigen::VectorXd::Zero(size);
			for ( int i = 0; i < size; i++ ) YtR(i) = Ys[i].cwiseProduct(Rvec).sum();
			const Eigen::VectorXd Gamma = YtYinv * YtR;
			if (output){
				std::printf("Extrapolation coefficients:");
				for ( int i = 0; i < size; i++ ) std::printf(" %f", Gamma(i));
				std::printf("\n");
			}

			// Obtaining the next step
			Eigen::VectorXd Rvec_bar = Rvec;
			for ( int i = 0; i < size; i++ ) Rvec_bar -= Ys[i] * Gamma(i);
			S = beta * Rvec_bar;
			for ( int i = 0; i < size; i++ ) S -= Ss[i] * Gamma(i);
		}else S = Rvec;

		const Eigen::VectorXd Pvec = M.Retract(S);
		DecoupleBlock(Pvec, P, M.BlockParameters);

		// Elapsed time
		if (output) std::printf("Elapsed time: %f seconds for current iteration; %f seconds in total\n\n", __duration__(iter_start, __now__), __duration__(all_start, __now__));
	}

	return converged;
}

#ifdef __PYTHON__
void Init_Anderson(pybind11::module_& m){
	m.def("Anderson", &Anderson);
}
#endif

}
