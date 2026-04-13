#include <Eigen/Dense>
#include <vector>
#include <deque>
#include <memory>

#include "../Macro.h"
#include "../Manifold/Manifold.h"

namespace Maniverse{

auto LagrangeAnderson(
		std::vector<double> tol, int max_mem, int max_iter, double beta,
		int output){ return [tol, max_mem, max_iter, beta, output](auto&& func){ return [tol, max_mem, max_iter, beta, output, func = std::forward<decltype<func>>(func)](auto&&... args) -> decltype(auto){
	const int ncons = (int)M.Func->Lambda.size();
	if ( output ){
		std::printf("************ Lagrange multiplier method with Anderson acceleration ************\n\n");
		std::printf("Number of constraints: %d\n", ncons);
		std::printf("Maximum number of iterations: %d\n", max_iter);
		std::printf("Maximum memory of previous iterations: %d\n", max_mem);
		std::printf("Tolerance of violation of constraints:");
		for ( int i = 0; i < ncons; i++ ) std::printf(" %f", tol[i]);
		std::printf("\n\n");
	}

	Iterate& M = std::get<0>(std::forward_as_tuple(args...));
	Eigen::Map<EigenVector> Lambda(M.Func->Lambda.data(), ncons);
	Eigen::Map<EigenVector> Value(M.Func->Constraint_Value.data(), ncons);
	std::deque<EigenVector> Updates, Errors;
	for ( int iiter = 0; iiter < max_iter; iiter++ ){
		if ( iiter == 0 ) Lambda.fill(1);
		if ( output ){
			std::printf("Iteration %d\n", iiter);
			std::printf("Lagrange multipliers:");
			for ( int i = 0; i < ncons; i++ ) std::printf(" %f", Lambda(i));
			std::printf("\n");
			std::printf("Running internal optimization ...\n");
		}
		if ( iiter == 0 ){
			M.Func->Calculate(M.getPoint(), {0, 1});
			M.setGradient();
		}else{
			const bool inner_converged = func(std::forward<decltype(args)>(args)...);
			if ( ! inner_converged ) throw std::runtime_error("Internal optimization did not converge!");
		}

		if ( output ){
			std::printf("Violation of constraints:");
			for ( int i = 0; i < ncons; i++ ) std::printf(" %f", Value(i));
			std::printf("\n");
		}
		for ( int i = 0 ; i < ncons; i++ ) if ( std::abs(Value(i)) > tol[i] ) goto NotConverged;
		if ( output ) std::printf("Converged!");
		return 1;

		NotConverged:
		if ( output ) std::printf("Not converged yet!");
		EigenVector Gf = M.Gradient;
		EigenMatrix Gg = EigenZero(Gl.size(), ncons);
		for ( int i = 0; i < ncons; i++ ){
			M.Func->Gradient = M.Func->Constraint_Gradient[i];
			M.setGradient();
			Gf -= M.Gradient;
			Gg.col(i) = M.Gradient;
		}
		const EigenVector update = - Gg.colPivHouseholderQr().solve(Gf);

		if ( iiter == 0 ){
			Lambda = 0.5 * ( Lambda + update );
		}else{
			const int current_size = (int)Lambdas.size();
			if ( current_size == max_mem ){
				Lambdas.pop_front();
				Errors.pop_front();
			}
			Updates.push_back(update);
			Errors.push_back(Value);
			EigenMatrix A = EigenZero(current_size + 1, current_size + 1);
			A.topRightCorner(current_size, 1).fill(1);
			A.bottomLeftCorner(1, current_size).fill(1);
			for ( int a = 0; a < current_size; a++ ){
				for ( int b = a; b < current_size; b++ ){
					A(a, b) = A(b, a) = Errors[a].cwiseProduct(Errors[b]).sum();
				}
			}
			EigenVector B = EigenZero(current_size + 1, 1);
			B(current_size) = 1;
			const EigenVector coeffs = - A.colPivHouseholderQr().solve(B);
			if ( output ){
				std::printf("Anderson coefficients:");
				for ( int i = 0; i < current_size; i++ ) std::printf(" %f", coeffs[i]);
				std::printf("\n");
			}
			Lambda = 0;
			for ( int i = 0; i < current_size; i++ ) Lambda += coeffs[i] * Updates[i];
		}
	}
	return 0;
};};}

