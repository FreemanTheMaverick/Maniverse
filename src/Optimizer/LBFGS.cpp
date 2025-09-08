#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#endif
#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <tuple>
#include <deque>
#include <cstdio>
#include <chrono>
#include <string>
#include <memory>

#include "../Macro.h"
#include "../Manifold/Manifold.h"
#include "LBFGS.h"

namespace Maniverse{

// https://doi.org/10.1016/j.procs.2016.05.534
template <typename FuncType>
bool LBFGS(
		FuncType& func,
		std::tuple<double, double, double> tol,
		int max_mem, int max_iter,
		double& L, Iterate& M, int output){

	auto [tol0, tol1, tol2] = tol;
	if (output > 0){
		std::printf("*************** Limited-Memory Broyden–Fletcher–Goldfarb–Shanno ***************\n\n");
		std::printf("Manifold: %s\n", M.getName().c_str());
		std::printf("Dimension number: %d\n", M.getDimension());
		if constexpr (std::is_same_v<FuncType, UnpreconFirstFunc>){
			std::printf("Preconditioner: False\n");
		}else if constexpr (std::is_same_v<FuncType, PreconFunc>){
			std::printf("Preconditioner: True\n");
		}
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
	std::vector<EigenMatrix> Ge;
	[[maybe_unused]] std::vector<std::function<EigenMatrix (EigenMatrix)>> Precon;
	[[maybe_unused]] std::vector<std::function<EigenMatrix (EigenMatrix)>> InvPrecon; // These variables may or may not be used, depending on whether UnpreconFirstFunc or PreconFunc is specified.

	std::deque<EigenMatrix> Ss;
	std::deque<EigenMatrix> Gs;
	std::deque<double> Rhos;
	double gamma = 1;

	bool converged = 0;

	for ( int iiter = 0; ( iiter < max_iter ) && ( ! converged ); iiter++ ){
		if (output) std::printf("Iteration %d\n", iiter);
		const auto iter_start = __now__;

		if constexpr (std::is_same_v<FuncType, UnpreconFirstFunc>){
			std::tie(L, Ge) = func(P, 1);
		}else if constexpr (std::is_same_v<FuncType, PreconFunc>){
			std::tie(L, Ge, Precon, InvPrecon) = func(P, 1);
		}
		actual_delta_L = L - oldL;
		oldL = L;
		if (output) std::printf("Target = %.10f\n", L);

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
		M.setGradient(Ge);
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
		if constexpr (std::is_same_v<FuncType, PreconFunc>){
			M.setPreconditioner(Precon);
			M.setInversePreconditioner(InvPrecon);
		}
		if ( iiter > 0 ){
			if constexpr (std::is_same_v<FuncType, UnpreconFirstFunc>){
				for ( int i = 0; i < (int)Ss.size(); i++ ){
					preconSs[i] = Ss[i];
					if ( i < (int)Ss.size() - 1 )
						preconYs[i] = Gs[i + 1] - Gs[i];
					else
						preconYs[i] = M.Gradient - Gs[i];
				}
			}else if constexpr (std::is_same_v<FuncType, PreconFunc>){
				for ( int i = 0; i < (int)Ss.size(); i++ ){
					preconSs[i] = M.InversePreconditioner(Ss[i]);
					if ( i < (int)Ss.size() - 1 )
						preconYs[i] = M.Preconditioner(Gs[i + 1]) - M.Preconditioner(Gs[i]);
					else
						preconYs[i] = M.Preconditioner(M.Gradient) - M.Preconditioner(Gs[i]);
				}
			}
			Rhos.push_back( 1. / M.Inner(preconSs.back(), preconYs.back())) ;
			gamma = 1. / Rhos.back() / M.Inner(preconYs.back(), preconYs.back());
		}

		// Obtaining the next step via L-BFGS
		if ( ! converged ){
			EigenMatrix Q = EigenZero(Pmat.rows(), Pmat.cols());
			if constexpr (std::is_same_v<FuncType, UnpreconFirstFunc>){
				Q = M.Gradient;
			}else if constexpr (std::is_same_v<FuncType, PreconFunc>){
				Q = M.Preconditioner(M.Gradient);
			}
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
			EigenMatrix Eta;
			if constexpr (std::is_same_v<FuncType, UnpreconFirstFunc>){
				Eta = - R;
			}else if constexpr (std::is_same_v<FuncType, PreconFunc>){
				Eta = - M.Preconditioner(R);
			}
			// TODO: Line search
			S = Eta;
			Snorm = std::sqrt(M.Inner(S, S));
			Pmat = M.Retract(S);
			DecoupleBlock(Pmat, P);
		}

		// Elapsed time
		if (output) std::printf("Elapsed time: %f seconds for current iteration; %f seconds in total\n\n", __duration__(iter_start, __now__), __duration__(all_start, __now__));
	}

	return converged;
}

template bool LBFGS(
		UnpreconFirstFunc& func,
		std::tuple<double, double, double> tol,
		int max_iter, int max_mem,
		double& L, Iterate& M, int output);

template bool LBFGS(
		PreconFunc& func,
		std::tuple<double, double, double> tol,
		int max_iter, int max_mem,
		double& L, Iterate& M, int output);

#ifdef __PYTHON__
void Init_LBFGS(pybind11::module_& m){
	m.def("LBFGS", &LBFGS<UnpreconFirstFunc>);
	m.def("PreconLBFGS", &LBFGS<PreconFunc>);
}
#endif

}
