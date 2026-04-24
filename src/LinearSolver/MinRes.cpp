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

#include "MinRes.h"

namespace Maniverse{

void MinRes::Calculate(double R){
	if (Verbose){
		std::printf("MinRes optimizer on the tangent space of %s manifold\n", M->getName().c_str());
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
	Eigen::VectorXd s = p;
	Eigen::VectorXd p_1 = p;
	Eigen::VectorXd s_1 = s;
	Eigen::VectorXd p_2 = p;
	Eigen::VectorXd s_2 = s;

	double vplusnorm = 0;
	double L = 0;
	double Llast = 0;
	const auto start = __now__;

	Eigen::VectorXd Ap = Eigen::VectorXd::Zero(M->TotalSize);
	Eigen::VectorXd vplus = Eigen::VectorXd::Zero(M->TotalSize);

	for ( int iiter = 0; iiter < M->getDimension(); iiter++ ){
		if (Verbose) std::printf("| %4d |", iiter);
		p_2 = p_1; s_2 = s_1;
		p_1 = p; s_1 = s;
		z = M->TangentPurification(P(r));
		p = s;
		s = M->TangentPurification(A(s));
		if ( iiter > 0 ){
			const double beta = M->Inner(s, s_1) / M->Inner(s_1, s_1);
			p = z - beta * p_1;
			s -= beta * s_1;
		}
		if ( iiter > 1 ){
			const double beta = M->Inner(s, s_2) / M->Inner(s_2, s_2);
			p -= beta * p_2;
			s -= beta * s_2;
		}
		if ( iiter % FuncFreq == FuncFreq - 1 ){
			Llast = L;
			L = 0.5 * M->Inner(A(v), v) + M->Inner(-b, v);
		}
		const double rnorm = std::sqrt(M->Inner(r, z));
		if (Verbose) std::printf("  %17.10f  | % 5.1E | %5.1E |", L, L - Llast, rnorm);
		const double alpha = M->Inner(r, s) / M->Inner(s, s);
		vplus = M->TangentPurification(v + alpha * p);
		vplusnorm = std::sqrt(M->Inner(vplus, vplus));
		const double step = std::abs(alpha) * std::sqrt(M->Inner(p, p));
		if (Verbose) std::printf(" %5.1E | %6.3f |\n", step, __duration__(start, __now__));
		if ( iiter > 0 && ( std::abs((L - Llast)/L) / FuncFreq < std::get<0>(Tolerance) || rnorm / std::sqrt(M->Inner(b, b)) < std::get<1>(Tolerance) ) ){
			if (Verbose) std::printf("Tolerance met!\n");
			Sequence.push_back(std::make_tuple(v, p));
			return;
		}

		if ( vplusnorm >= R ){
			if (Verbose && vplusnorm >= R) std::printf("Out of trust region!\n");
			Sequence.push_back(SteihaugToint(v, p, R));
			return;
		}

		v = vplus;
		Sequence.push_back(std::make_tuple(v, p));
		r -= alpha * s;
	}
	if (Verbose) std::printf("Dimension completed!\n");
}

#ifdef __PYTHON__
void Init_MinRes(pybind11::module_& m){
	pybind11::classh<MinRes, LinearSolver>(m, "MinRes")
		.def(pybind11::init<
				int, std::tuple<double, double>, bool
		>());
}
#endif

}
