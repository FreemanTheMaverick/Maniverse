#include <Eigen/Dense>
#include <vector>
#include <deque>
#include <memory>

#include "../Macro.h"
#include "../Manifold/Manifold.h"

namespace Maniverse{

auto Lagrange(
		double init_rho, double theta_rho, double theta_sigma,
		std::vector<double> tol, int max_iter, int output){ return [init_rho, theta_rho, theta_sigma, tol, max_iter, output](auto&& func){ return [init_rho, theta_rho, theta_sigma, tol, max_iter, output, func = std::forward<decltype<func>>(func)](auto&&... args) -> decltype(auto){
	const int ncons = (int)M.Func->Lambda.size();
	if ( output ){
		std::printf("***************************** Augmented Lagrangian *****************************\n\n");
		std::printf("Number of constraints: %d\n", ncons);
		std::printf("Maximum number of iterations: %d\n", max_iter);
		std::printf("Tolerance of constraint violation:");
		for ( int i = 0; i < ncons; i++ ) std::printf(" %f", tol[i]);
		std::printf("\n\n");
	}

	Iterate& M = std::get<0>(std::forward_as_tuple(args...));
	double& Rho = M.Func->Rho;
	Eigen::Map<EigenVector> Lambda(M.Func->Lambda.data(), ncons);
	Eigen::Map<EigenVector> Violation(M.Func->Constraint_Value.data(), ncons);
	EigenVector Violation_old = Violation;
	EigenVector Lambda_min = Lambda;
	EigenVector Lambda_max = Lambda;
	for ( int iiter = 0; iiter < max_iter; iiter++ ){
		if ( iiter == 0 ){
			Lambda.fill(1);
			Rho = 0;
		}
		if ( output ){
			std::printf("Iteration %d\n", iiter);
			std::printf("Lagrange multipliers:");
			for ( int i = 0; i < ncons; i++ ) std::printf(" %f", Lambda(i));
			std::printf("\n");
			std::printf("Penalty factor: %f\n", Rho);
			std::printf("Running internal optimization ...\n");
		}
		if ( iiter == 0 ){
			M.Func->Calculate(M.getPoint(), {0, 1});
			M.setGradient();
			EigenVector Gf = M.Gradient;
			EigenMatrix Gg = EigenZero(Gl.size(), ncons);
			for ( int i = 0; i < ncons; i++ ){
				M.Func->Gradient = M.Func->Constraint_Gradient[i];
				M.setGradient();
				Gf -= M.Gradient;
				Gg.col(i) = M.Gradient;
			}
			Lambda = - Gg.colPivHouseholderQr().solve(Gf);
			Rho = init_rho;
		}else{
			const bool inner_converged = func(std::forward<decltype(args)>(args)...);
			if ( ! inner_converged ) throw std::runtime_error("Internal optimization did not converge!");
		}

		if ( output ){
			std::printf("Constraint violation:");
			for ( int i = 0; i < ncons; i++ ) std::printf(" %f", Violation(i));
			std::printf("\n");
		}
		for ( int i = 0 ; i < ncons; i++ ) if ( std::abs(Violation(i)) > tol[i] ) goto NotConverged;
		if ( output ) std::printf("Converged!");
		return 1;

		NotConverged:
		if ( output ) std::printf("Not converged yet!");
		if ( iiter > 0 ){
			for ( int i = 0; i < ncons; i++ ){
				Lambda(i) += Rho * Violation(i);
			}
			if ( Violation.cwiseAbs().maxCoeff() > theta_sigma * Violation_old.cwiseAbs().maxCoeff() ) Rho *= theta_rho;
		}
		Violation_old = Violation;
	}
	return 0;
};};}

