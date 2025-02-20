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
#include "TrustRegion.h"
#include "SubSolver.h"
#include "HessUpdate.h"

#include <iostream>


TrustRegionSetting::TrustRegionSetting(){
	this->R0 = 1;
	this->RhoThreshold = 0.1;
	this->Update = [&R0 = this->R0, &RhoThreshold = this->RhoThreshold](double R, double Rho, double S2){
		if ( Rho < 0.25 ) R *= 0.25;
		else if ( Rho > 0.75 || std::abs(S2 - R * R) < 1.e-10 ) R = std::min(2 * R, R0);
		return R;
	};
}

bool TrustRegion(
		std::function<
			std::tuple<
				double,
				EigenMatrix,
				std::function<EigenMatrix (EigenMatrix)>
			> (EigenMatrix, int)
		>& func,
		TrustRegionSetting& tr_setting,
		std::tuple<double, double, double> tol,
		int recalc_hess, int max_iter,
		double& L, Manifold& M, int output){

	const double tol0 = std::get<0>(tol) * M.getDimension();
	const double tol1 = std::get<1>(tol) * M.P.size();
	const double tol2 = std::get<2>(tol) * M.P.size();
	if (output > 0){
		std::printf("*********************** Trust Region Optimizer Vanilla ************************\n\n");
		std::printf("Manifold: %s\n", M.Name.c_str());
		std::printf("Matrix free: %s\n", __True_False__(M.MatrixFree));
		std::printf("Maximum number of iterations: %d\n", max_iter);
		std::printf("True hessian calculated every %d iterations\n", recalc_hess);
		std::printf("Trust region settings:\n");
		std::printf("| Initial radius: %f\n", tr_setting.R0);
		std::printf("| Rho threshold: %f\n", tr_setting.RhoThreshold);
		std::printf("Convergence threshold:\n");
		std::printf("| Target change (T. C.)               : %E\n", tol0);
		std::printf("| Gradient norm (Grad.)               : %E\n", tol1);
		std::printf("| Independent variable update (V. U.) : %E\n\n", tol2);
	}

	const auto all_start = __now__;

	double R = tr_setting.R0;
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
	bool converged = 0;

	for ( int iiter = 0; ( iiter < max_iter ) && ( ! converged ); iiter++ ){
		if (output) std::printf("Iteration %d\n", iiter);
		const auto iter_start = __now__;

		const bool calc_hess = iiter == 0 || (int)Ms.size() == recalc_hess;
		if (output) std::printf("Calculate true hessian: %s\n", __True_False__(calc_hess));

		std::tie(L, Ge, He) = func(P, calc_hess ? 2 : 1);

		// Rating the new step
		actual_delta_L = L - oldL;
		const double rho = actual_delta_L / predicted_delta_L;
		if ( ( accepted = ( rho > tr_setting.RhoThreshold || iiter == 0 ) ) ){
			oldL = L;
			M.Update(P, 1);
			M.Ge = Ge;
			M.He = He;
		}
		if (output){
			std::printf("Target = %.10f\n", L);
			std::printf("Actual change = %E; Predicted change = %E\n", actual_delta_L, predicted_delta_L);
			if (accepted) std::printf("Score of the new step Rho = %f, compared with RhoThreshold %f. Step accepted.\n", rho, tr_setting.RhoThreshold);
			else std::printf("Score of the new step Rho = %f, compared with RhoThreshold %f. Step rejected.\n", rho, tr_setting.RhoThreshold);
		}

		// Obtaining Riemannian gradient
		M.getGradient();
		const double Gnorm = std::sqrt(std::abs(M.Inner(M.Gr, M.Gr)));

		// Checking convergence
		if (output){
			std::printf("Convergence info: current / threshold / converged?\n");
			std::printf("| Target    change: % E / %E / %s\n", actual_delta_L, tol0, __True_False__(std::abs(actual_delta_L) < tol0));
			std::printf("| Gradient    norm: % E / %E / %s\n", Gnorm, tol1, __True_False__(Gnorm < tol1));
			std::printf("| Step length norm: % E / %E / %s\n", std::sqrt(S2), tol2, __True_False__(std::sqrt(S2) < tol2));
			if ( std::abs(actual_delta_L) < tol0 && Gnorm < tol1 && std::sqrt(S2) < tol2 ) std::printf("| Converged!\n");
			else std::printf("| Not converged yet!\n");
		}
		if ( Gnorm < tol1 ){
			if ( iiter == 0 ) converged = 1;
			else if ( std::abs(actual_delta_L) < tol0 && std::sqrt(S2) < tol2 ) converged = 1;
		}

		// Adjusting the trust radius according to the score
		if ( ! converged ){
			if ( iiter > 0 ) R = tr_setting.Update(R, rho, S2);
			if (output) std::printf("Trust radius is adjusted to %f\n", R);
		}

		// Preparing hessian
		if (accepted && ( ! converged )){
			if ( ! M.MatrixFree ) M.getBasisSet();
			if (calc_hess){
				Ms.clear();
				M.getHessian();
				if ( ! M.MatrixFree ) M.getHessianMatrix();
			}else BroydenFletcherGoldfarbShanno(*(Ms.back()), M, S);

			if ( ! M.MatrixFree ){
				int negative = 0;
				for ( auto& [eigenvalue, _] : M.Hrm ){
					if ( eigenvalue < 0 ) negative++;
				}
				const double shift = std::get<0>(M.Hrm[negative]) - std::get<0>(M.Hrm[0]);
				for ( auto& [eigenvalue, _] : M.Hrm ) eigenvalue += shift;
				if (output){
					std::printf("Hessian has %d negative eigenvalues\n", negative);
					std::printf("Lowest eigenvalue is %f\n", std::get<0>(M.Hrm[0]) - shift);
					if (negative) std::printf("All eigenvalues will be shifted up by %f\n", shift);
				}
			}

			// Truncated conjugate gradient
			const std::tuple<double, double, double> tcg_tol = {
				tol0/M.getDimension(),
				0.1*std::min(M.Inner(M.Gr,M.Gr),std::sqrt(M.Inner(M.Gr,M.Gr)))/M.getDimension(),
				0.1*tol2/M.getDimension()
			};
			Ss = TruncatedConjugateGradient(M, R, tcg_tol, output-1);
			Ms.push_back(M.Clone());
		}

		// Obtaining the next step within the trust region
		if ( ! converged ){
			S = RestartTCG(M, Ss, R); // "RestartTCG" is supposed to give the step that is the most compatible with the trust radius.
			S2 = M.Inner(S, S);
			P = M.Exponential(S);
			predicted_delta_L = M.Inner(M.Gr + 0.5 * M.Hr(S), S);
			if (output){
				std::printf("Next step:\n");
				std::printf("| Step length: %E\n", std::sqrt(S2));
				std::printf("| Predicted change in target: %E\n", predicted_delta_L);
			}
		}

		// Elapsed time
		if (output) std::printf("Elapsed time: %f seconds for current iteration; %f seconds in total\n\n", __duration__(iter_start, __now__), __duration__(all_start, __now__));
	}

	return 0;
}

bool TrustRegionRationalFunction(
		std::function<
			std::tuple<
				double,
				EigenMatrix,
				std::function<EigenMatrix (EigenMatrix)>
			> (EigenMatrix, int)
		>& func,
		TrustRegionSetting& tr_setting,
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

	double R = tr_setting.R0;
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
		if ( ( accepted = ( rho > tr_setting.RhoThreshold || iiter == 0 ) ) ){
			oldL = L;
			M.Update(P, 1);
			M.Ge = Ge;
			M.He = He;
		}

		// Adjusting the trust radius according to the score
		if ( iiter > 0 ) R = tr_setting.Update(R, rho, S2);

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
			if ( ! M.MatrixFree ) M.getBasisSet();
			if (calc_hess){
				Ms.clear();
				M.getHessian();
				if ( ! M.MatrixFree ) M.getHessianMatrix();
			}else BroydenFletcherGoldfarbShanno(*(Ms.back()), M, S);

			if ( ! M.MatrixFree ){
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
			Ms.push_back(M.Clone());
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
	pybind11::class_<TrustRegionSetting>(m, "TrustRegionSetting")
		.def_readwrite("R0", &TrustRegionSetting::R0)
		.def_readwrite("RhoThreshold", &TrustRegionSetting::RhoThreshold)
		.def_readwrite("Update", &TrustRegionSetting::Update)
		.def(pybind11::init<>());
	m.def("TrustRegion", &TrustRegion);
	m.def("TrustRegionRationalFunction", &TrustRegionRationalFunction);
}
