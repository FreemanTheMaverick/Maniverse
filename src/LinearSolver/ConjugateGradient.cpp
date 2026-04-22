#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#endif

#include <Eigen/Dense>
#include <cstdio>
#include <cmath>
#include <tuple>
#include <chrono>

#include "../Macro.h"

#include "ConjugateGradient.h"

namespace Maniverse{

void ConjugateGradient::Calculate(double R){
	if (Verbose){
		std::printf("Conjugated gradient optimizer on the tangent space of %s manifold\n", M->getName().c_str());
		std::printf("Frown at non-positive curvature: %d\n", FrownNPC);
		std::printf("Frequency of quadratic function evaluation: %d\n", FuncFreq);
		std::printf("Tolerance of relative quadratic lowering: %E\n", std::get<0>(Tolerance));
		std::printf("Tolerance of residual                   : %E\n", std::get<1>(Tolerance));
		std::printf("Maximal search radius: %f\n", R);
		std::printf("| Itn. |       Target        |   Diff.  |  Grad.  |  V. U.  |  Time  |\n");
	}

	Sequence.clear(); Sequence.reserve(20);
	Eigen::VectorXd v = Eigen::VectorXd::Zero(M->TotalSize);
	Eigen::VectorXd r = b;
	Eigen::VectorXd z = P(r);
	Eigen::VectorXd p = z;

	double vplusnorm = 0;
	double r2 = M->Inner(r, z);
	double L = 0;
	double Llast = 0;
	const auto start = __now__;

	Eigen::VectorXd Ap = Eigen::VectorXd::Zero(M->TotalSize);
	Eigen::VectorXd vplus = Eigen::VectorXd::Zero(M->TotalSize);

	for ( int iiter = 0; iiter < M->getDimension(); iiter++ ){
		if (Verbose) std::printf("| %4d |", iiter);
		Ap = M->TangentPurification(A(p));
		const double pAp = M->Inner(p, Ap);
		if ( iiter % FuncFreq == FuncFreq - 1 ){
			Llast = L;
			L = 0.5 * M->Inner(A(v), v) + M->Inner(-b, v);
		}
		if (Verbose) std::printf("  %17.10f  | % 5.1E | %5.1E |", L, L - Llast, std::sqrt(r2));
		const double alpha = r2 / pAp;
		vplus = M->TangentPurification(v + alpha * p);
		vplusnorm = std::sqrt(M->Inner(vplus, vplus));
		const double step = std::abs(alpha) * std::sqrt(M->Inner(p, p));
		if (Verbose) std::printf(" %5.1E | %6.3f |\n", step, __duration__(start, __now__));
		if ( iiter > 0 && ( std::abs((L - Llast)/L) / FuncFreq < std::get<0>(Tolerance) || std::sqrt(r2) < std::get<1>(Tolerance) ) ){
			if (Verbose) std::printf("Tolerance met!\n");
			Sequence.push_back(std::make_tuple(v, p));
			return;
		}

		if ( ( FrownNPC && pAp <= 0 ) || vplusnorm >= R ){
			if (Verbose && pAp <= 0) std::printf("Non-positive curvature!\n");
			if (Verbose && vplusnorm >= R) std::printf("Out of trust region!\n");
			Sequence.push_back(SteihaugToint(v, p, R));
			return;
		}

		v = vplus;
		Sequence.push_back(std::make_tuple(v, p));
		const double r2old = r2;
		r -= alpha * Ap;
		const Eigen::MatrixXd z = M->TangentPurification(P(r));
		r2 = M->Inner(r, z);
		const double beta = r2 / r2old;
		p = z + beta * p;
	}
	if (Verbose) std::printf("Dimension completed!\n");
}

#ifdef __PYTHON__
void Init_ConjugateGradient(pybind11::module_& m){
	pybind11::classh<ConjugateGradient, LinearSolver>(m, "ConjugateGradient").
		.def(pybind11::init<
				int, bool, bool, std::tuple<double, double>
		>());
}
#endif

}
