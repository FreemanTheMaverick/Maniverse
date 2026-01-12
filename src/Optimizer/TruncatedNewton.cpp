#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#endif
#include <Eigen/Dense>
#include <cmath>
#include <tuple>
#include <map>
#include <cstdio>
#include <chrono>
#include <string>
#include <memory>

#include "../Macro.h"
#include "../Manifold/Manifold.h"
#include "TrustRegion.h"
#include "TruncatedNewton.h"

namespace Maniverse{

#define G M->Gradient

void TruncatedConjugateGradient::Run(){
	if (this->Verbose){
		std::printf("Using truncated conjugated gradient optimizer on the tangent space of %s manifold\n", this->M->getName().c_str());
		std::printf("| Itn. |       Target        |   T. C.  |  Grad.  |  V. U.  |  Time  |\n");
	}

	this->Sequence.clear(); this->Sequence.reserve(20);
	EigenMatrix v = EigenZero(G.rows(), G.cols());
	EigenMatrix r = - G;
	EigenMatrix z = this->M->Preconditioner(r);
	EigenMatrix p = z;
	double vnorm = 0;
	double vplusnorm = 0;
	double r2 = this->M->Inner(r, z);
	double L = 0;
	const auto start = __now__;

	EigenMatrix Hp = EigenZero(G.rows(), G.cols());
	EigenMatrix vplus = EigenZero(G.rows(), G.cols());

	for ( int iiter = 0; iiter < this->M->getDimension(); iiter++ ){
		if (this->Verbose) std::printf("| %4d |", iiter);
		Hp = this->M->TangentPurification(this->M->Hessian(p));
		const double pHp = this->M->Inner(p, Hp);
		const double Llast = L;
		if (this->ShowTarget) L = 0.5 * this->M->Inner(this->M->Hessian(v), v) + this->M->Inner(G, v);
		else L = std::nan("");
		const double deltaL = L - Llast;
		if (this->Verbose) std::printf("  %17.10f  | % 5.1E | %5.1E |", L, deltaL, std::sqrt(r2));

		const double alpha = r2 / pHp;
		vplus = this->M->TangentPurification(v + alpha * p);
		vplusnorm = std::sqrt(this->M->Inner(vplus, vplus));
		vnorm = std::sqrt(this->M->Inner(v, v));
		const double step = std::abs(alpha) * std::sqrt(this->M->Inner(p, p));
		if (this->Verbose) std::printf(" %5.1E | %6.3f |\n", step, __duration__(start, __now__));
		if ( iiter > 0 && this->Tolerance(deltaL, L, std::sqrt(r2), step) ){
			if (this->Verbose) std::printf("Tolerance met!\n");
			this->Sequence.push_back(std::make_tuple(vnorm, v, p));
			return;
		}

		if ( pHp <= 0 || vplusnorm >= this->Radius ){
			const double A = this->M->Inner(p, p);
			const double B = this->M->Inner(v, p) * 2.;
			const double C = vnorm * vnorm - this->Radius * this->Radius;
			const double t = ( std::sqrt( B * B - 4. * A * C ) - B ) / 2. / A;
			if (this->Verbose && pHp <= 0) std::printf("Non-positive curvature!\n");
			if (this->Verbose && vplusnorm >= this->Radius) std::printf("Out of trust region!\n");
			this->Sequence.push_back(std::make_tuple(this->Radius, v + t * p, p));
			return;
		}
		v = vplus;
		vnorm = vplusnorm;
		this->Sequence.push_back(std::make_tuple(vnorm, v, p));
		const double r2old = r2;
		r -= alpha * Hp;
		const EigenMatrix z = this->M->TangentPurification(this->M->Preconditioner(r));
		r2 = this->M->Inner(r, z);
		const double beta = r2 / r2old;
		p = z + beta * p;
	}
	if (this->Verbose) std::printf("Dimension completed!\n");
}

std::tuple<double, EigenMatrix> TruncatedConjugateGradient::Find(){
	for ( int i = 0; i < (int)this->Sequence.size(); i++ ) if ( std::get<0>(this->Sequence[i]) > this->Radius ){
		const EigenMatrix v = std::get<1>(this->Sequence[i]);
		const EigenMatrix p = std::get<2>(this->Sequence[i]);
		const double A = this->M->Inner(p, p);
		const double B = this->M->Inner(v, p) * 2.;
		const double C = this->M->Inner(v, v) - this->Radius * this->Radius;
		const double t = ( std::sqrt( B * B - 4. * A * C ) - B ) / 2. / A;
		const EigenMatrix vnew = v + t * p;
		return std::make_tuple(this->Radius, vnew);
	}
	return std::make_tuple(
			std::get<0>(this->Sequence.back()),
			std::get<1>(this->Sequence.back())
	);
}

bool TruncatedNewton(
		Iterate& M,
		TrustRegion& tr,
		std::tuple<double, double, double> tol,
		double tcg_tol, int max_iter,
		int output){

	auto [tol0, tol1, tol2] = tol;
	if (output > 0){
		std::printf("*********************** Trust Region Optimizer Vanilla ************************\n\n");
		std::printf("Manifold: %s\n", M.getName().c_str());
		std::printf("Dimension number: %d\n", M.getDimension());
		std::printf("Matrix free: %s\n", __True_False__(M.MatrixFree));
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

	TruncatedConjugateGradient tcg{&M, output > 0, 1};

	EigenMatrix Pmat = M.Point;
	EigenMatrix S = EigenZero(Pmat.rows(), Pmat.cols());
	double Snorm = 0;
	double Gnorm = 0;
	std::vector<EigenMatrix> P = M.getPoint();

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
			tcg.Radius = R;
			if ( iiter > 0 ) std::tie(Snorm, S) = tcg.Find();
			Pmat = M.Retract(S);
			DecoupleBlock(Pmat, P);
			if ( iiter > 0 ) predicted_delta_L = M.Inner(M.Gradient + 0.5 * M.Hessian(S), S);
			if (output){
				std::printf("Trial %d - %d:\n", iiter, trial);
				std::printf("Step length: %E\n", Snorm);
				std::printf("Predicted change in target: %E\n", predicted_delta_L);
			}

			// Evaluating the objective function
			M.Func->Calculate(P, 2);

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
			tcg.Tolerance = [tcg_tol](double deltaL, double L, double /*rnorm*/, double /*step*/){
				return std::abs(deltaL / L) < tcg_tol;
			};
			tcg.Radius = R;
			tcg.Run();
		}

		// Elapsed time
		if (output) std::printf("Elapsed time: %f seconds for current iteration; %f seconds in total\n\n", __duration__(iter_start, __now__), __duration__(all_start, __now__));
	}

	return converged;
}

#ifdef __PYTHON__
void Init_TruncatedNewton(pybind11::module_& m){
	pybind11::class_<TruncatedConjugateGradient>(m, "TruncatedConjugateGradient")
		.def_readwrite("M", &TruncatedConjugateGradient::M)
		.def_readwrite("Verbose", &TruncatedConjugateGradient::Verbose)
		.def_readwrite("ShowTarget", &TruncatedConjugateGradient::ShowTarget)
		.def_readwrite("Radius", &TruncatedConjugateGradient::Radius)
		.def_readwrite("Tolerance", &TruncatedConjugateGradient::Tolerance)
		.def_readwrite("Sequence", &TruncatedConjugateGradient::Sequence)
		.def(pybind11::init<>())
		.def(pybind11::init<Iterate*, bool, bool>())
		.def("Run", &TruncatedConjugateGradient::Run)
		.def("Find", &TruncatedConjugateGradient::Find);
	m.def("TruncatedNewton", &TruncatedNewton);
}
#endif

}
