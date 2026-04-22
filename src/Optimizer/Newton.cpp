#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#endif

#include <Eigen/Dense>
#include <cmath>
#include <tuple>
#include <cstdio>
#include <chrono>

#include "../Macro.h"
#include "../Manifold/Manifold.h"
#include "../LinearSolver/LinearSolver.h"

#include "TrustRegion.h"
#include "Newton.h"

namespace Maniverse{

bool Newton(
		Iterate& M,
		TrustRegion& tr,
		LinearSolver& ls,
		std::tuple<double, double, double> tol,
		int max_iter, int output){

	auto [tol0, tol1, tol2] = tol;
	if (output > 0){
		std::printf("******************************* Newton's method ********************************\n\n");
		std::printf("Manifold: %s\n", M.getName().c_str());
		std::printf("Dimension number: %d\n", M.getDimension());
		std::printf("Maximum number of iterations: %d\n", max_iter);
		std::printf("Trust region settings:\n");
		std::printf("| Initial radius: %f\n", tr.R0);
		std::printf("| Rho threshold: %f\n", tr.RhoThreshold);
		std::printf("Convergence threshold:\n");
		std::printf("| Target change (T. C.)               : %E\n", tol0);
		std::printf("| Gradient norm (Grad.)               : %E\n", tol1);
		std::printf("| Independent variable update (V. U.) : %E\n\n", tol2);
	}

	const auto all_start = __now__;

	double R = tr.R0;

	double oldL = 0;
	double predicted_delta_L = 0;
	double actual_delta_L = 0;

	Eigen::MatrixXd Pmat = M.Point;
	Eigen::MatrixXd S = Eigen::MatrixXd::Zero(Pmat.rows(), Pmat.cols());
	double Snorm = 0;
	double Gnorm = 0;
	std::vector<Eigen::MatrixXd> P = M.getPoint();

	bool converged = 0;
	for ( int iiter = 0; ( iiter < max_iter ) && ( ! converged ); iiter++ ){
		if (output){
			std::printf("Iteration %d\n", iiter);
			std::printf("---------------------------------------------------------------\n");
		}

		const auto iter_start = __now__;

		bool accepted = 0;
		for ( int trial = 0; ! accepted; trial++ ){

			// Obtaining the next step within the trust region
			if ( iiter > 0 ) S = ls.Find(R);
			Snorm = M.Inner(S, S);
			Pmat = M.Retract(S);
			DecoupleBlock(Pmat, P, M.BlockParameters);
			if ( iiter > 0 ) predicted_delta_L = M.Inner(M.Gradient + 0.5 * M.Hessian(S), S);
			if (output){
				std::printf("Trial %d - %d:\n", iiter, trial);
				std::printf("Step length: %E\n", Snorm);
				std::printf("Predicted change in target: %E\n", predicted_delta_L);
			}

			// Evaluating the objective function
			M.Func->Calculate(P, {0});

			// Rating the new step
			actual_delta_L = M.Func->Value - oldL;
			const double rho = actual_delta_L / predicted_delta_L;
			accepted = ( rho > tr.RhoThreshold || iiter == 0 || ( Gnorm < tol1 && Snorm < tol2 ) );
			if (output){
				std::printf("Target = %.10f\n", M.Func->Value);
				std::printf("Step score:\n");
				std::printf("| Predicted and actual changes in target = %E, %E\n", predicted_delta_L, actual_delta_L);
				std::printf("| Score of the new step Rho = %f, compared with RhoThreshold %f\n", rho, tr.RhoThreshold);
				if (accepted) std::printf("| Step accepted\n");
				else std::printf("| Step rejected\n");
			}

			// Adjusting the trust radius according to the score
			if ( iiter > 0 ) R = tr.Update(R, rho, Snorm);
			if (output){
				std::printf("Trust radius is adjusted to %f\n", R);
				std::printf("---------------------------------------------------------------\n");
			}
		}

		// Evaluating the Euclidean derivatives
		M.Func->Calculate(P, {1, 2});

		// Updating the new step
		oldL = M.Func->Value;
		M.setPoint(P, 1);

		// Obtaining Riemannian gradient
		M.setGradient();
		Gnorm = std::sqrt(std::abs(M.Inner(M.Gradient, M.Gradient)));

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

		// Preparing hessian and storing this step
		if ( ! converged ){
			// Truncated conjugate gradient
			ls.M = &M;
			ls.A = [&M](Eigen::VectorXd X){ return M.Hessian(X); };
			ls.b = - M.Gradient;
			ls.P = [&M](Eigen::VectorXd X){ return M.Preconditioner(X); };
			ls.Calculate(R);
		}

		// Elapsed time
		if (output) std::printf("Elapsed time: %f seconds for current iteration; %f seconds in total\n\n", __duration__(iter_start, __now__), __duration__(all_start, __now__));
	}

	return converged;
}

#ifdef __PYTHON__
void Init_Newton(pybind11::module_& m){
	m.def("Newton", &Newton);
}
#endif

}
