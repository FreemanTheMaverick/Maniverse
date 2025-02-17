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

#include "../Macro.h"
#include "../Manifold/Manifold.h"
#include "SubSolver.h"

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
	std::tie(L, M.Ge, M.He) = func(M.P, 2);
	double deltaL = L;
	double R = R0;
	
	for ( int iiter = 0; iiter < max_iter; iiter++ ){

		M.getGradient();
		if (output > 0) std::printf("| %4d |  %17.10f  | % 5.1E | %5.1E |", iiter, L, deltaL, M.Gr.norm());
		if (__Calc_Hess__(iiter)) M.getHessian();

		if ( ! M.MatrixFree ){
			M.getBasisSet();
			M.getHessianMatrix();
			int negative = 0;
			for ( auto& [eigenvalue, _] : M.Hrm ){
				if ( eigenvalue < 0 ) negative++;
			}
			double shift = 0;
			if ( negative > 0 ) shift = std::get<0>(M.Hrm[negative]) - std::get<0>(M.Hrm[0]);
			for ( auto& [eigenvalue, _] : M.Hrm ) eigenvalue += shift;
		}

		// Truncated conjugate gradient and rating the new step
		const std::tuple<double, double, double> loong_tol = {
			tol0/M.getDimension(),
			0.1*std::min(M.Inner(M.Gr,M.Gr),std::sqrt(M.Inner(M.Gr,M.Gr)))/M.getDimension(),
			0.1*tol2/M.getDimension()
		};
		const EigenMatrix S = TruncatedConjugateGradient(M, R, loong_tol, output-1);

		const double S2 = M.Inner(S, S);
		const EigenMatrix Pnew = M.Exponential(S);
		double Lnew;
		EigenMatrix Genew;
		std::function<EigenMatrix (EigenMatrix)> Henew;
		std::tie(Lnew, Genew, Henew) = func(Pnew, __Calc_Hess__(iiter) ? 2 : 1);
		const double top = Lnew - L;
		const double bottom = M.Inner(M.Gr + 0.5 * M.Hr(S), S);
		const double rho = top / bottom;

		// Determining whether to accept or reject the step
		if ( rho > rho_thres ){
			deltaL = Lnew - L;
			L = Lnew;
			M.Update(Pnew, 1);
			M.Ge = Genew;
			M.He = Henew;
			if (output > 0) std::printf(" Accept |");
		}else if (output > 0) std::printf(" Reject |");
		if (output > 0) std::printf(" %5.1E | %6.3f |\n", std::sqrt(S2), __duration__(start, __now__));

		// Adjusting the trust radius according to the score
		if ( rho < 0.25 ) R *= 0.25;
		else if ( rho > 0.75 || std::abs(S2 - R * R) < 1.e-10 ) R = std::min(2 * R, R0);
		if ( M.Gr.norm() < tol1 ){
			if ( iiter == 0 ){
				if ( std::sqrt(S2) < tol2 ) return 1;
			}else{
				if ( std::abs(deltaL) < tol0 && std::sqrt(S2) < tol2 ) return 1;
			}
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
		int order,
		std::tuple<double, double, double> tol,
		int max_iter,
		double& L, Manifold& M, int output){

	const double tol0 = std::get<0>(tol) * M.getDimension();
	const double tol1 = std::get<1>(tol) * M.P.size();
	const double tol2 = std::get<2>(tol) * M.P.size();
	if (output > 0){
		std::printf("Using trust region rational function optimizer on %s manifold\n", M.Name.c_str());
		std::printf("Convergence threshold:\n");
		std::printf("| Target change (T. C.)               : %E\n", tol0);
		std::printf("| Gradient norm (Grad.)               : %E\n", tol1);
		std::printf("| Independent variable update (V. U.) : %E\n", tol2);
		std::printf("| Itn. |       Target        |   T. C.  |  Grad.  | Update |  V. U.  |  Time  |\n");
	}

	const auto start = __now__;
	const double R0 = 1;
	const double rho_thres = 0.1;
	std::tie(L, M.Ge, M.He) = func(M.P, 2);
	double deltaL = L;
	double R = R0;
	
	for ( int iiter = 0; iiter < max_iter; iiter++ ){

		M.getGradient();
		if (output > 0) std::printf("| %4d |  %17.10f  | % 5.1E | %5.1E |", iiter, L, deltaL, M.Gr.norm());
		M.getHessian();
		M.getBasisSet();
		M.getHessianMatrix();
		int negative = 0;
		for ( auto& [eigenvalue, _] : M.Hrm ){
			if ( eigenvalue < 0 ) negative++;
		}
		if ( order != negative ){
			double shift = 0;
			if ( order == 0 ) shift = std::get<0>(M.Hrm[negative]) - std::get<0>(M.Hrm[0]);
			else shift = - ( std::get<0>(M.Hrm[order - 1]) + std::get<0>(M.Hrm[order]) ) / 2;
			for ( auto& [eigenvalue, _] : M.Hrm ) eigenvalue += shift;
		}

		// Truncated conjugate gradient and rating the new step
		const std::tuple<double, double, double> loong_tol = {
			tol0/M.getDimension(),
			0.1*std::min(M.Inner(M.Gr,M.Gr),std::sqrt(M.Inner(M.Gr,M.Gr)))/M.getDimension(),
			0.1*tol2/M.getDimension()
		};
		const EigenMatrix S = TruncatedConjugateGradient(M, R, loong_tol, output-1);

		const double S2 = M.Inner(S, S);
		const EigenMatrix Pnew = M.Exponential(S);
		double Lnew;
		EigenMatrix Genew;
		std::function<EigenMatrix (EigenMatrix)> Henew;
		std::tie(Lnew, Genew, Henew) = func(Pnew, 2);
		const double top = Lnew - L;
		const double bottom = M.Inner(M.Gr + 0.5 * M.Hr(S), S);
		const double rho = top / bottom;

		// Determining whether to accept or reject the step
		if ( rho > rho_thres ){
			deltaL = Lnew - L;
			L = Lnew;
			M.Update(Pnew, 1);
			M.Ge = Genew;
			M.He = Henew;
			if (output > 0) std::printf(" Accept |");
		}else if (output > 0) std::printf(" Reject |");
		if (output > 0) std::printf(" %5.1E | %6.3f |\n", std::sqrt(S2), __duration__(start, __now__));

		// Adjusting the trust radius according to the score
		if ( rho < 0.25 ) R *= 0.25;
		else if ( rho > 0.75 || std::abs(S2 - R * R) < 1.e-10 ) R = std::min(2 * R, R0);
		if ( M.Gr.norm() < tol1 ){
			if ( iiter == 0 ){
				if ( std::sqrt(S2) < tol2 ) return 1;
			}else{
				if ( std::abs(deltaL) < tol0 && std::sqrt(S2) < tol2 ) return 1;
			}
		}

	}

	return 0;
}

void Init_TrustRegion(pybind11::module_& m){
	m.def("TrustRegion", &TrustRegion);
	m.def("TrustRegionRationalFunctionOptimization", &TrustRegionRationalFunctionOptimization);
}
