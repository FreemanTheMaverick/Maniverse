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
pybind11::function AugmentedLagrangian(
		double init_rho, double theta_rho, double theta_sigma,
		std::vector<double> tol, int max_iter, int output){ return pybind11::cpp_function([init_rho, theta_rho, theta_sigma, tol, max_iter, output](pybind11::function func) -> pybind11::cpp_function{ return pybind11::cpp_function([init_rho, theta_rho, theta_sigma, tol, max_iter, output, func](pybind11::args args, pybind11::kwargs kwargs) -> bool{
#else
auto AugmentedLagrangian(
		double init_rho, double theta_rho, double theta_sigma,
		std::vector<double> tol, int max_iter, int output){ return [init_rho, theta_rho, theta_sigma, tol, max_iter, output](auto&& func){ return [init_rho, theta_rho, theta_sigma, tol, max_iter, output, func = std::forward<decltype(func)>(func)](auto&&... args) -> bool{
#endif
	#ifdef __PYTHON__
	Iterate& M = args[0].cast<Iterate&>();
	#else
	Iterate& M = std::get<0>(std::forward_as_tuple(args...));
	#endif
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
			#ifdef __PYTHON__
			const bool inner_converged = pybind11::bool_(func(*args, **kwargs));
			#else
			const bool inner_converged = func(std::forward<decltype(args)>(args)...);
			#endif
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
		return true;

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
	return false;
#ifdef __PYTHON__
});});}
#else
};};}
#endif

}
