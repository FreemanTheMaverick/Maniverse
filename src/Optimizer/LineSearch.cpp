#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#endif
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <tuple>
#include <functional>
#include <cstdio>
#include <chrono>
#include <cassert>
#include <string>
#include <tuple>
#include <memory>

#include "../Macro.h"
#include "../Manifold/Manifold.h"
#include "LineSearch.h"

namespace Maniverse{

template <typename FuncType>
bool ArmijoBacktracking(
		FuncType& func,
		Iterate& M, EigenMatrix& S,
		double c1, double tau, int max_iter,
		double& L, std::vector<EigenMatrix>* Ge,
		[[maybe_unused]] std::vector<std::function<EigenMatrix (EigenMatrix)>>* func1,
		[[maybe_unused]] std::vector<std::function<EigenMatrix (EigenMatrix)>>* func2,
		int output){

	if ( output > 0 ){
		std::printf("Armijo backtracking line search on the tangent space of %s manifold\n", M.getName().c_str());
		std::printf("Armijo's C1: %f\n", c1);
		std::printf("Shrinking rate tau: %f\n", tau);
		std::printf("| Itn. |   Ratio   |   Step   |      Target       |  Target - L.H.S.  |  Time  |\n");
	}

	const double oldL = L;
	const double SGr = M.Inner(S, M.Gradient);
	const double Snorm = std::sqrt(M.Inner(S, S));

	double alpha = 1;
	std::vector<EigenMatrix> P = M.getPoint();
	for ( int iiter = 0; iiter < max_iter; iiter++ ){
		const auto start = __now__;
		const EigenMatrix Pmat = M.Retract(alpha * S);
		DecoupleBlock(Pmat, P);
		if constexpr (std::is_same_v<FuncType, UnpreconFirstFunc>){
			std::tie(L, *Ge) = func(P, 1);
		}else if constexpr (std::is_same_v<FuncType, UnpreconSecondFunc>){
			std::tie(L, *Ge, *func1) = func(P, 1);
		}else if constexpr (std::is_same_v<FuncType, PreconFunc>){
			std::tie(L, *Ge, *func1, *func2) = func(P, 1);
		}

		const double RHS = oldL + c1 * alpha * SGr;

		if ( output > 0 ) std::printf("| %4d |  %5.2E | %5.2E |  %15.10f  |        % 5.2E  | %6.3f |\n", iiter, alpha, alpha * Snorm, L, L - RHS, __duration__(start, __now__));
		if ( L <= RHS ){
			S *= alpha;
			return 1;
		}
		alpha *= tau;
	}
	return 0;
}

template bool ArmijoBacktracking(
		UnpreconFirstFunc& func,
		Iterate& M, EigenMatrix& S,
		double c1, double tau, int max_iter,
		double& L, std::vector<EigenMatrix>* Ge,
		[[maybe_unused]] std::vector<std::function<EigenMatrix (EigenMatrix)>>* func1,
		[[maybe_unused]]std::vector<std::function<EigenMatrix (EigenMatrix)>>* func2,
		int output);

template bool ArmijoBacktracking(
		UnpreconSecondFunc& func,
		Iterate& M, EigenMatrix& S,
		double c1, double tau, int max_iter,
		double& L, std::vector<EigenMatrix>* Ge,
		[[maybe_unused]] std::vector<std::function<EigenMatrix (EigenMatrix)>>* func1,
		[[maybe_unused]] std::vector<std::function<EigenMatrix (EigenMatrix)>>* func2,
		int output);

template bool ArmijoBacktracking(
		PreconFunc& func,
		Iterate& M, EigenMatrix& S,
		double c1, double tau, int max_iter,
		double& L, std::vector<EigenMatrix>* Ge,
		[[maybe_unused]] std::vector<std::function<EigenMatrix (EigenMatrix)>>* func1,
		[[maybe_unused]] std::vector<std::function<EigenMatrix (EigenMatrix)>>* func2,
		int output);

#ifdef __PYTHON__
void Init_LineSearch(pybind11::module_& m){
	m.def("UnpreconFirstArmijoBacktracking", &ArmijoBacktracking<UnpreconFirstFunc>);
	m.def("UnpreconSecondArmijoBacktracking", &ArmijoBacktracking<UnpreconSecondFunc>);
	m.def("PreconArmijoBacktracking", &ArmijoBacktracking<PreconFunc>);
}
#endif

}
