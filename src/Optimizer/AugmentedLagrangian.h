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

#define __Print_Constraint_Status__ {\
	std::printf("Constraint violation:    ");\
	for ( Constraint& constraint : M.Constraints ) std::printf(" % E", constraint.Func->Value);\
	std::printf("\n");\
	std::printf("Constraint gradient norm:");\
	for ( Constraint& constraint : M.Constraints ) std::printf(" % E", constraint.Gradient.norm());\
	std::printf("\n");\
	Eigen::MatrixXd cons_jac(M.Point.size(), ncons);\
	for ( int i = 0; i < (int)M.Constraints.size(); i++ ) cons_jac.col(i) = M.Constraints[i].Gradient;\
	Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(cons_jac);\
	const double smallest = qr.matrixQR().diagonal().cwiseAbs().minCoeff();\
	std::printf("Linear dependence: %E\n", smallest);\
}

#ifdef __PYTHON__
pybind11::function AugmentedLagrangian(
		double init_rho, double theta_rho, double theta_sigma,
		std::vector<double> tol, int max_iter, int output){ return pybind11::cpp_function([=](pybind11::function func) -> pybind11::cpp_function{ return pybind11::cpp_function([=](pybind11::args args, pybind11::kwargs kwargs) -> bool{
#else
static auto AugmentedLagrangian(
		double init_rho, double theta_rho, double theta_sigma,
		std::vector<double> tol, int max_iter, int output){ return [=](auto&& func){ return [=, func = std::forward<decltype(func)>(func)](auto&&... args) -> bool{
#endif
	#ifdef __PYTHON__
	Iterate& M = args[0].cast<Iterate&>();
	#else
	Iterate& M = std::get<0>(std::forward_as_tuple(args...));
	#endif
	const int ncons = (int)M.Constraints.size();
	if ( output ){
		std::printf("***************************** Augmented Lagrangian *****************************\n\n");
		std::printf("Number of constraints: %d\n", ncons);
		std::printf("Maximum number of iterations: %d\n", max_iter);
		std::printf("Tolerance of constraint violation:");
		for ( int i = 0; i < ncons; i++ ) std::printf(" %E", tol[i]);
		std::printf("\n");
	}

	double& Rho = M.Rho = 0;
	double last_max_vio = 0;

	if ( output ) std::printf("First run for the initial multipliers ...\n");
	M.Calculate(M.getPoint(), {0, 1});
	M.setGradient();
	if ( output){ __Print_Constraint_Status__ }
	M.setLambda(M.calcLambda());
	Rho = init_rho;

	for ( int iiter = 0; iiter < max_iter; iiter++ ){
		if ( output ){
			std::printf("\nIteration %d\n", iiter);
			std::printf("Lagrange multipliers:");
			for ( Constraint& constraint : M.Constraints ) std::printf(" %f", constraint.Lambda);
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

		if ( output){ __Print_Constraint_Status__ }
		for ( int i = 0 ; i < ncons; i++ ) if ( std::abs(M.Constraints[i].Func->Value) > tol[i] ) goto NotConverged;
		if ( output ){
			std::printf("Converged!\n");
			std::printf("Final Lagrange multipliers:");
			for ( Constraint& constraint : M.Constraints ) std::printf(" %f", constraint.Lambda);
			std::printf("\n");
		}
		return true;

		NotConverged:
		if ( output ) std::printf("Not converged yet!\n");
		const double max_vio = std::abs(std::max_element(M.Constraints.begin(), M.Constraints.end(), [](const Constraint& a, const Constraint& b){ return std::abs(a.Func->Value) < std::abs(b.Func->Value); })->Func->Value);
		for ( int i = 0; i < ncons; i++ ){
			M.Constraints[i].Lambda += Rho * M.Constraints[i].Func->Value;
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
