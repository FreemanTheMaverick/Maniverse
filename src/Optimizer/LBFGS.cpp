#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#endif
#include <Eigen/Dense>
#include <cmath>
#include <tuple>
#include <deque>
#include <cstdio>
#include <chrono>
#include <string>
#include <memory>

#include "../Macro.h"
#include "../Manifold/Manifold.h"
#include "LBFGS.h"
#include "LineSearch.h"

namespace Maniverse{

// https://doi.org/10.1016/j.procs.2016.05.534
bool LBFGS(
		Iterate& M,
		std::tuple<double, double, double> tol,
		int max_mem, int max_iter,
		double c1, double tau, int ls_max_iter,
		int output){

	auto [tol0, tol1, tol2] = tol;
	if (output > 0){
		std::printf("*************** Limited-Memory Broyden–Fletcher–Goldfarb–Shanno ***************\n\n");
		std::printf("Manifold: %s\n", M.getName().c_str());
		std::printf("Dimension number: %d\n", M.getDimension());
		std::printf("Maximum number of iterations: %d\n", max_iter);
		std::printf("Maximum memory of previous iterations: %d\n", max_mem);
		std::printf("Convergence threshold:\n");
		std::printf("| Target change (T. C.)               : %E\n", tol0);
		std::printf("| Gradient norm (Grad.)               : %E\n", tol1);
		std::printf("| Independent variable update (V. U.) : %E\n\n", tol2);
	}

	const auto all_start = __now__;

	double oldL = 0;
	double actual_delta_L = 0;

	EigenMatrix Pmat = M.Point;
	EigenMatrix S = EigenZero(Pmat.rows(), Pmat.cols());
	double Snorm = 0;
	std::vector<EigenMatrix> P = M.getPoint();

	std::deque<EigenMatrix> Ss;
	std::deque<EigenMatrix> Gs;
	std::deque<double> Rhos;
	double gamma = 1;

	bool converged = 0;

	for ( int iiter = 0; ( iiter < max_iter ) && ( ! converged ); iiter++ ){
		if (output) std::printf("Iteration %d\n", iiter);
		const auto iter_start = __now__;

		ArmijoBacktracking(
			M, S,
			c1, tau, iiter == 0 ? 1 : ls_max_iter,
			output > 0
		);

		actual_delta_L = M.Func->Value - oldL;
		oldL = M.Func->Value;
		if (output) std::printf("Target = %.10f\n", M.Func->Value);

		Snorm = std::sqrt(M.Inner(S, S));
		Pmat = M.Retract(S);
		DecoupleBlock(Pmat, P);

		// Transporting previous vectors I
		if ( (int)Rhos.size() == max_mem ){
			Ss.pop_front();
			Gs.pop_front();
			Rhos.pop_front();
		}
		for ( int i = 0 ; i < (int)Rhos.size(); i++ ){
			Ss[i] = M.TransportTangent(Ss[i], S);
			Gs[i] = M.TransportTangent(Gs[i], S);
		}
		if ( iiter > 0 ){
			Ss.push_back(M.TransportTangent(S, S));
			Gs.push_back(M.TransportTangent(M.Gradient, S));
		}

		// Obtaining Riemannian gradient
		M.setPoint(P, 1);
		M.setGradient();
		const double Gnorm = std::sqrt(std::abs(M.Inner(M.Gradient, M.Gradient)));

		// Checking convergence
		if ( Gnorm < tol1 ){
			if ( iiter == 0 ) converged = 1;
			else if ( std::abs(actual_delta_L) < tol0 && Snorm < tol2 ) converged = 1;
		}
		if (output){
			std::printf("Convergence info: current / threshold / converged?\n");
			std::printf("| Target    change: % E / %E / %s\n", actual_delta_L, tol0, __True_False__(std::abs(actual_delta_L) < tol0));
			std::printf("| Gradient    norm: % E / %E / %s\n", Gnorm, tol1, __True_False__(Gnorm < tol1));
			std::printf("| Step length norm: % E / %E / %s\n", Snorm, tol2, __True_False__(Snorm < tol2));
			if ( converged ) std::printf("| Converged!\n");
			else std::printf("| Not converged yet!\n");
		}

		// Transporting previous vectors II
		std::vector<EigenMatrix> preconSs(Ss.size(), EigenZero(Pmat.rows(), Pmat.cols()));
		std::vector<EigenMatrix> preconYs(Ss.size(), EigenZero(Pmat.rows(), Pmat.cols()));
		if ( iiter > 0 ){
			for ( int i = 0; i < (int)Ss.size(); i++ ){
				preconSs[i] = M.PreconditionerInvSqrt(Ss[i]);
				if ( i < (int)Ss.size() - 1 )
					preconYs[i] = M.PreconditionerSqrt(Gs[i + 1]) - M.PreconditionerSqrt(Gs[i]);
				else
					preconYs[i] = M.PreconditionerSqrt(M.Gradient) - M.PreconditionerSqrt(Gs[i]);
			}
			Rhos.push_back( 1. / M.Inner(preconSs.back(), preconYs.back())) ;
			gamma = 1. / Rhos.back() / M.Inner(preconYs.back(), preconYs.back());
		}

		// Obtaining the next step via L-BFGS
		if ( ! converged ){
			EigenMatrix Q = M.PreconditionerSqrt(M.Gradient);
			const int mem = (int)Ss.size();
			if ( output > 0 ) std::printf("Current memory size: %d\n", mem);
			std::vector<double> Ksis(mem);
			for ( int i = (int)mem - 1; i >= 0; i-- ){
				Ksis[i] = Rhos[i] * M.Inner(preconSs[i], Q);
				Q -= Ksis[i] * preconYs[i];
			}
			EigenMatrix R = gamma * Q;
			for ( int i = 0; i < mem; i++ ){
				const double omega = Rhos[i] * M.Inner(preconYs[i], R);
				R += preconSs[i] * ( Ksis[i] - omega );
			}
			EigenMatrix Eta = - M.PreconditionerSqrt(R);
			S = Eta;
		}

		// Elapsed time
		if (output) std::printf("Elapsed time: %f seconds for current iteration; %f seconds in total\n\n", __duration__(iter_start, __now__), __duration__(all_start, __now__));
	}

	return converged;
}

#ifdef __PYTHON__
void Init_LBFGS(pybind11::module_& m){
	m.def("LBFGS", &LBFGS);
}
#endif

}
