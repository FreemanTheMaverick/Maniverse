#pragma once

#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>
#endif

#include <Eigen/Dense>
#include <vector>
#include <deque>
#include <memory>

#ifdef __PYTHON__
#include "../Macro.h"
#include "../Manifold/Manifold.h"
#else
#include <Maniverse/Manifold/Manifold.h>
#endif

namespace Maniverse{

#ifdef __PYTHON__
#define FuncType pybind11::function
#define FuncDef func
#define FuncArgs pybind11::args args, pybind11::kwargs kwargs
#define FuncFirstArg args[0].cast<Iterate&>()
#define FuncCall pybind11::bool_(func(*args, **kwargs))
#else 
#define FuncType auto&&
#define FuncDef func = std::forward<decltype(func)>(func)
#define FuncArgs auto&&... args
#define FuncFirstArg std::get<0>(std::forward_as_tuple(args...))
#define FuncCall func(std::forward<decltype(args)>(args)...)
#endif

auto AugmentedLagrangian(
		double init_rho, double theta_rho, double theta_sigma,
		std::vector<double> tol, int max_iter, int output){ return [init_rho, theta_rho, theta_sigma, tol, max_iter, output](FuncType func){ return [init_rho, theta_rho, theta_sigma, tol, max_iter, output, FuncDef](FuncArgs) -> bool{
	Iterate& M = FuncFirstArg;
	const int ncons = (int)M.Func->Lambda.size();
	if ( output ){
		std::printf("***************************** Augmented Lagrangian *****************************\n\n");
		std::printf("Number of constraints: %d\n", ncons);
		std::printf("Maximum number of iterations: %d\n", max_iter);
		std::printf("Tolerance of constraint violation:");
		for ( int i = 0; i < ncons; i++ ) std::printf(" %E", tol[i]);
		std::printf("\n");
	}

	double& Rho = M.Func->Rho;
	double last_max_vio = 0;
	for ( int iiter = 0; iiter < max_iter; iiter++ ){
		std::vector<double>& Lambda = M.Func->Lambda;
		std::vector<double>& Violation = M.Func->Constraint_Value;
		if ( iiter == 0 ){
			std::memset(Lambda.data(), 0, ncons * 8);
			Rho = 0;
		}
		if ( output ){
			std::printf("\nIteration %d\n", iiter);
			std::printf("Lagrange multipliers:");
			for ( int i = 0; i < ncons; i++ ) std::printf(" %f", Lambda[i]);
			std::printf("\n");
			std::printf("Penalty factor: %f\n", Rho);
			std::printf("Running internal optimization ...\n");
		}
		if ( iiter == 0 ){
			M.Func->Calculate(M.getPoint(), {0, 1});
			M.setGradient();
			const EigenVector Gf = M.Gradient;
			EigenMatrix Gg = Eigen::MatrixXd::Zero(Gf.size(), ncons);
			for ( int i = 0; i < ncons; i++ ){
				M.Func->Gradient = M.Func->Constraint_Gradient[i];
				M.setGradient();
				Gg.col(i) = M.Gradient;
			}
			const Eigen::VectorXd tmp = - Gg.colPivHouseholderQr().solve(Gf);
			std::memcpy(Lambda.data(), tmp.data(), ncons * 8);
			Rho = init_rho;
		}else{
			const bool inner_converged = FuncCall;
			if ( ! inner_converged ) throw std::runtime_error("Internal optimization did not converge!");
		}

		if ( output ){
			std::printf("Constraint violation:");
			for ( int i = 0; i < ncons; i++ ) std::printf(" %E", Violation[i]);
			std::printf("\n");
		}
		if ( iiter == 0 ) goto NotConverged;
		for ( int i = 0 ; i < ncons; i++ ) if ( std::abs(Violation[i]) > tol[i] ) goto NotConverged;
		if ( output ) std::printf("Converged!\n");
		return 1;

		NotConverged:
		if ( output ) std::printf("Not converged yet!\n");
		const double max_vio = *std::max_element(Violation.begin(), Violation.end(), [](const int& a, const int& b){ return abs(a) < abs(b); });
		if ( iiter > 0 ){
			for ( int i = 0; i < ncons; i++ ){
				Lambda[i] += Rho * Violation[i];
			}
		}
		if ( iiter > 1 ){
			if ( max_vio > theta_sigma * last_max_vio ) Rho *= theta_rho;
		}
		last_max_vio = max_vio;
	}
	return 0;
};};}

#undef FuncType
#undef FuncDef
#undef FuncArgs
#undef FuncReturn
#undef FuncFirstArg
#undef FuncCall

}
