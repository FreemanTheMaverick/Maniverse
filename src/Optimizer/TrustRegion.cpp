#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <tuple>
#include <cstdio>
#include <chrono>
#include <string>
#include <memory>

#include "../Macro.h"
#include "../Manifold/Manifold.h"
#include "SubSolver.h"
#include "HessUpdate.h"

#include <iostream>


#define __Calc_Hess__(n) ( n % recalc_hess == 0 )

bool TrustRegion(
		std::function<
			std::tuple<
				double,
				EigenMatrix,
				std::function<EigenMatrix (EigenMatrix)>
			> (EigenMatrix, int)
		>& func,
		std::tuple<double, double, double> tol,
		int recalc_hess, int max_iter,
		double& L, Manifold& M, int output){

	const double tol0 = std::get<0>(tol) * M.getDimension();
	const double tol1 = std::get<1>(tol) * M.P.size();
	const double tol2 = std::get<2>(tol) * M.P.size();
	if (output > 0){
		std::printf("Using trust region optimizer on %s manifold\n", M.Name.c_str());
		std::printf("Convergence threshold:\n");
		std::printf("| Target change (T. C.)               : %E\n", tol0);
		std::printf("| Gradient norm (Grad.)               : %E\n", tol1);
		std::printf("| Independent variable update (V. U.) : %E\n", tol2);
		std::printf("| Itn. |       Target        |   T. C.  |  Grad.  | Update |  V. U.  |  Time  |\n");
	}

	const auto start = __now__;

	const double R0 = 1;
	const double rho_thres = 0.1;
	double R = R0;
	double oldL = 0;
	double actual_delta_L = 0;
	double predicted_delta_L = 0;

	std::vector<std::unique_ptr<Manifold>> Ms; Ms.reserve(recalc_hess);
	std::vector<std::tuple<double, EigenMatrix, EigenMatrix>> Ss;
	EigenMatrix S = EigenZero(M.P.rows(), M.P.cols());
	double S2 = 0;
	EigenMatrix P = M.P;
	EigenMatrix Ge = EigenZero(P.rows(), P.cols());
	std::function<EigenMatrix (EigenMatrix)> He = [](EigenMatrix v){return v;};
	bool accepted = 1;

	for ( int iiter = 0; iiter < max_iter; iiter++ ){

		const bool calc_hess = iiter == 0 || (int)Ms.size() == recalc_hess;

		std::tie(L, Ge, He) = func(P, calc_hess ? 2 : 1);

		// Scoring the new step
		actual_delta_L = L - oldL;
		const double rho = actual_delta_L / predicted_delta_L;
		if ( ( accepted = ( rho > rho_thres || iiter == 0 ) ) ){
			oldL = L;
			M.Update(P, 1);
			M.Ge = Ge;
			M.He = He;
			Ms.push_back(M.Clone());
		}

		// Adjusting the trust radius according to the score
		if ( iiter > 0 ){
			if ( rho < 0.25 ) R *= 0.25;
			else if ( rho > 0.75 || std::abs(S2 - R * R) < 1.e-10 ) R = std::min(2 * R, R0);
		}

		// Obtaining Riemannian gradient
		M.getGradient();

		// Checking convergence
		if ( M.Gr.norm() < tol1 ){
			if ( iiter == 0 ){
				if ( std::sqrt(S2) < tol2 ) return 1;
			}else{
				if ( std::abs(actual_delta_L) < tol0 && std::sqrt(S2) < tol2 ) return 1;
			}
		}

		if (accepted){
			if (calc_hess){
				Ms.clear();
				M.getHessian();
			}else BroydenFletcherGoldfarbShanno(*(Ms.back()), M, S);

			if ( ! M.MatrixFree ){
				M.getBasisSet();
				M.getHessianMatrix();
				int negative = 0;
				for ( auto& [eigenvalue, _] : M.Hrm ){
					if ( eigenvalue < 0 ) negative++;
				}
				const double shift = std::get<0>(M.Hrm[negative]) - std::get<0>(M.Hrm[0]);
				for ( auto& [eigenvalue, _] : M.Hrm ) eigenvalue += shift;
			}

			// Truncated conjugate gradient
			const std::tuple<double, double, double> tcg_tol = {
				tol0/M.getDimension(),
				0.1*std::min(M.Inner(M.Gr,M.Gr),std::sqrt(M.Inner(M.Gr,M.Gr)))/M.getDimension(),
				0.1*tol2/M.getDimension()
			};
			Ss = TruncatedConjugateGradient(M, R, tcg_tol, output-1);
		}

		// Obtaining the new step within the trust region
		S = RestartTCG(M, Ss, R); // "RestartTCG" is supposed to give the step that is the most compatible with the trust radius.
		S2 = M.Inner(S, S);
		P = M.Exponential(S);
		predicted_delta_L = M.Inner(M.Gr + 0.5 * M.Hr(S), S);

		// Determining whether to accept or reject the step
		if (output > 0){
			if (accepted) std::printf("| %4d |  %17.10f  | % 5.1E | %5.1E | Accept | %5.1E | %6.3f |\n", iiter, L, actual_delta_L, M.Gr.norm(), std::sqrt(S2), __duration__(start, __now__));
			else std::printf("| %4d |  %17.10f  | % 5.1E | %5.1E | Reject | %5.1E | %6.3f |\n", iiter, L, actual_delta_L, M.Gr.norm(), std::sqrt(S2), __duration__(start, __now__));
		}
	}

	return 0;
}

bool TrustRegionRationalFunctionOptimization(
		std::function<
			std::tuple<
				double,
				EigenMatrix,
				std::function<EigenMatrix (EigenMatrix)>
			> (EigenMatrix, int)
		>& func,
		std::tuple<double, double, double> tol,
		int recalc_hess, int max_iter,
		double& L, Manifold& M, int output){

	const double tol0 = std::get<0>(tol) * M.getDimension();
	const double tol1 = std::get<1>(tol) * M.P.size();
	const double tol2 = std::get<2>(tol) * M.P.size();
	if (output > 0){
		std::printf("Using trust region optimizer on %s manifold\n", M.Name.c_str());
		std::printf("Convergence threshold:\n");
		std::printf("| Target change (T. C.)               : %E\n", tol0);
		std::printf("| Gradient norm (Grad.)               : %E\n", tol1);
		std::printf("| Independent variable update (V. U.) : %E\n", tol2);
		std::printf("| Itn. |       Target        |   T. C.  |  Grad.  | Update |  V. U.  |  Time  |\n");
	}

	const auto start = __now__;

	const double R0 = 1;
	const double rho_thres = 0.1;
	double R = R0;
	double oldL = 0;
	double actual_delta_L = 0;
	double predicted_delta_L = 0;

	std::vector<std::unique_ptr<Manifold>> Ms; Ms.reserve(recalc_hess);
	std::vector<std::tuple<double, EigenMatrix, EigenMatrix>> Ss;
	EigenMatrix S = EigenZero(M.P.rows(), M.P.cols());
	double S2 = 0;
	EigenMatrix P = M.P;
	EigenMatrix Ge = EigenZero(P.rows(), P.cols());
	std::function<EigenMatrix (EigenMatrix)> He = [](EigenMatrix v){return v;};
	bool accepted = 1;

	for ( int iiter = 0; iiter < max_iter; iiter++ ){

		const bool calc_hess = iiter == 0 || (int)Ms.size() == recalc_hess;

		std::tie(L, Ge, He) = func(P, calc_hess ? 2 : 1);

		// Scoring the new step
		actual_delta_L = L - oldL;
		const double rho = actual_delta_L / predicted_delta_L;
		if ( ( accepted = ( rho > rho_thres || iiter == 0 ) ) ){
			oldL = L;
			M.Update(P, 1);
			M.Ge = Ge;
			M.He = He;
			Ms.push_back(M.Clone());
		}

		// Adjusting the trust radius according to the score
		if ( iiter > 0 ){
			if ( rho < 0.25 ) R *= 0.25;
			else if ( rho > 0.75 || std::abs(S2 - R * R) < 1.e-10 ) R = std::min(2 * R, R0);
		}

		// Obtaining Riemannian gradient
		M.getGradient();

		// Checking convergence
		if ( M.Gr.norm() < tol1 ){
			if ( iiter == 0 ){
				if ( std::sqrt(S2) < tol2 ) return 1;
			}else{
				if ( std::abs(actual_delta_L) < tol0 && std::sqrt(S2) < tol2 ) return 1;
			}
		}

		if (accepted){
			if (calc_hess){
				Ms.clear();
				M.getHessian();
			}else BroydenFletcherGoldfarbShanno(*(Ms.back()), M, S);

			if ( ! M.MatrixFree ){
				M.getBasisSet();
				M.getHessianMatrix();
				int negative = 0;
				for ( auto& [eigenvalue, _] : M.Hrm ){
					if ( eigenvalue < 0 ) negative++;
				}
				const double shift = std::get<0>(M.Hrm[negative]) - std::get<0>(M.Hrm[0]);
				for ( auto& [eigenvalue, _] : M.Hrm ) eigenvalue += shift;
			}

			// Truncated conjugate gradient
			const std::tuple<double, double, double> tcg_tol = {
				tol0/M.getDimension(),
				0.1*std::min(M.Inner(M.Gr,M.Gr),std::sqrt(M.Inner(M.Gr,M.Gr)))/M.getDimension(),
				0.1*tol2/M.getDimension()
			};
			Ss = TruncatedConjugateGradient(M, R, tcg_tol, output-1);
		}

		// Obtaining the new step within the trust region
		S = RestartTCG(M, Ss, R); // "RestartTCG" is supposed to give the step that is the most compatible with the trust radius.
		S2 = M.Inner(S, S);
		P = M.Exponential(S);
		predicted_delta_L = M.Inner(M.Gr + 0.5 * M.Hr(S), S);

		// Determining whether to accept or reject the step
		if (output > 0){
			if (accepted) std::printf("| %4d |  %17.10f  | % 5.1E | %5.1E | Accept | %5.1E | %6.3f |\n", iiter, L, actual_delta_L, M.Gr.norm(), std::sqrt(S2), __duration__(start, __now__));
			else std::printf("| %4d |  %17.10f  | % 5.1E | %5.1E | Reject | %5.1E | %6.3f |\n", iiter, L, actual_delta_L, M.Gr.norm(), std::sqrt(S2), __duration__(start, __now__));
		}
	}

	return 0;
}

void Init_TrustRegion(pybind11::module_& m){
	m.def("TrustRegion", &TrustRegion);
	m.def("TrustRegionRationalFunctionOptimization", &TrustRegionRationalFunctionOptimization);
}
