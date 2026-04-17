#pragma once

#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>
#endif

#include <Eigen/Dense>
#include <vector>

#ifdef __PYTHON__
#include "../Manifold/Manifold.h"
#else
#include <Maniverse/Manifold/Manifold.h>
#endif

namespace Maniverse{

#ifdef __PYTHON__
pybind11::function AugmentedLagrangian(
		double init_rho, double theta_rho, double theta_sigma,
		std::vector<double> tol, int max_iter, int output){ return pybind11::cpp_function([=](pybind11::function func) -> pybind11::cpp_function{ return pybind11::cpp_function([=](pybind11::args args, pybind11::kwargs kwargs) -> bool{
#else
auto AugmentedLagrangian(
		double init_rho, double theta_rho, double theta_sigma,
		std::vector<double> tol, int max_iter, int output){ return [=](auto&& func){ return [=, func = std::forward<decltype(func)>(func)](auto&&... args) -> bool{
#endif
	#ifdef __PYTHON__
	Iterate& M = args[0].cast<Iterate&>();
	#else
	Iterate& M = std::get<0>(std::forward_as_tuple(args...));
	#endif
	std::vector<double>& Lambda = M.Func->Lambda;
	std::vector<double>& Violation = M.Func->Constraint_Value;
	const int ncons = (int)Lambda.size();
	if ( output ){
		std::printf("***************************** Augmented Lagrangian *****************************\n\n");
		std::printf("Number of constraints: %d\n", ncons);
		std::printf("Maximum number of iterations: %d\n", max_iter);
		std::printf("Tolerance of constraint violation:");
		for ( int i = 0; i < ncons; i++ ) std::printf(" %E", tol[i]);
		std::printf("\n");
	}

	double& Rho = M.Func->Rho = 0;
	std::memset(Lambda.data(), 0, ncons * 8);
	double last_max_vio = 0;

	if ( output ) std::printf("First run for the initial multipliers ...\n");
	M.Func->Calculate(M.getPoint(), {0, 1});
	M.setGradient();
	const Eigen::VectorXd Gf = M.Gradient;
	Eigen::MatrixXd Gg = Eigen::MatrixXd::Zero(Gf.size(), ncons);
	for ( int i = 0; i < ncons; i++ ){
		Gg.col(i) = M.Constraint_Gradient[i];
	}
	const Eigen::VectorXd tmp = - Gg.colPivHouseholderQr().solve(Gf);

	std::memcpy(Lambda.data(), tmp.data(), ncons * 8);
	Rho = init_rho;

	for ( int iiter = 0; iiter < max_iter; iiter++ ){
		if ( output ){
			std::printf("\nIteration %d\n", iiter);
			std::printf("Lagrange multipliers:");
			for ( int i = 0; i < ncons; i++ ) std::printf(" %f", Lambda[i]);
			std::printf("\n");
			std::printf("Penalty factor: %f\n", Rho);
			std::printf("Running internal optimization ...\n");
		}
		#ifdef __PYTHON__
		const bool inner_converged = pybind11::bool_(func(*args, **kwargs));
		#else
		const bool inner_converged = func(std::forward<decltype(args)>(args)...);
		#endif
		if ( ! inner_converged ) throw std::runtime_error("Internal optimization did not converge!");

		if ( output ){
			std::printf("Constraint violation:");
			for ( int i = 0; i < ncons; i++ ) std::printf(" %E", Violation[i]);
			std::printf("\n");
		}
		for ( int i = 0 ; i < ncons; i++ ) if ( std::abs(Violation[i]) > tol[i] ) goto NotConverged;
		if ( output ){
			std::printf("Converged!\n");
			std::printf("Final Lagrange multipliers:");
			for ( int i = 0 ; i < ncons; i++ ) std::printf(" %f", Lambda[i]);
			std::printf("\n");
		}
		return true;

		NotConverged:
		if ( output ) std::printf("Not converged yet!\n");
		const double max_vio = *std::max_element(Violation.begin(), Violation.end(), [](const int& a, const int& b){ return abs(a) < abs(b); });
		for ( int i = 0; i < ncons; i++ ){
			Lambda[i] += Rho * Violation[i];
		}
		if ( iiter > 0 && max_vio > theta_sigma * last_max_vio ) Rho *= theta_rho;
		last_max_vio = max_vio;
	}
	return false;
#ifdef __PYTHON__
});});}
#else
};};}
#endif

}
