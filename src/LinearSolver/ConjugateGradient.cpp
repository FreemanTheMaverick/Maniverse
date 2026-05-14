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

static double SteihaugToint(
		std::function<double (Eigen::VectorXd, Eigen::VectorXd)> dot,
		Eigen::VectorXd v, Eigen::VectorXd p, double R){
	const double A = dot(p, p);
	const double B = dot(v, p) * 2;
	const double C = dot(v, v) - R * R;
	const double t = ( std::sqrt( B * B - 4 * A * C ) - B ) / 2 / A;
	return t;
}

void ConjugateGradient::Calculate(double R){
	if (Verbose){
		std::printf("Linear system solving with conjugate gradient\n");
		std::printf("Maximal search radius          : %f\n", R);
		std::printf("Maximal number of iterations   : %d\n", MaxIter);
		std::printf("| Itn. |  Resi.  |       Target        |   Diff.  | Length  |  Time  |\n");
	}

	const int total_size = b.size();
	Sequence.clear(); Sequence.reserve(20);

	Eigen::VectorXd r = b;
	Eigen::VectorXd v = Eigen::VectorXd::Zero(total_size);
	Eigen::VectorXd p = Eigen::VectorXd::Zero(total_size);

	double r2 = 114514;
	double L = 0;
	const auto start = __now__;

	for ( int iiter = 0; iiter < MaxIter; iiter++ ){
		if (Verbose) std::printf("| %4d ", iiter);

		const Eigen::VectorXd z = proj(P(r));
		const double r2old = r2;
		r2 = dot(r, z);
		if (Verbose) std::printf("| %5.1E ", std::sqrt(r2));

		const double Llast = L;
		L = 0.5 * dot( r - b, v );
		if (Verbose) std::printf("|  %17.10f  | % 5.1E |", L, L - Llast);

		const double beta = r2 / r2old;
		p = z + beta * p;
		const Eigen::VectorXd Ap = proj(A(p));
		const double pAp = dot(p, Ap);
		const double alpha = r2 / pAp;
		const Eigen::VectorXd vplus = proj(v + alpha * p);
		const double vplusnorm = std::sqrt(dot(vplus, vplus));
		if (Verbose) std::printf(" %5.1E | %6.3f |\n", vplusnorm, __duration__(start, __now__));

		if ( ( FrownNPC && pAp <= 0 ) || vplusnorm >= R ){
			if (Verbose && FrownNPC && pAp <= 0) std::printf("Non-positive curvature!\n");
			if (Verbose && vplusnorm >= R) std::printf("Out of trust region!\n");
			const double t = SteihaugToint(dot, v, p, R);
			Sequence.push_back(std::make_tuple(v, t * p));
			return;
		}

		Sequence.push_back(std::make_tuple(v, alpha * p));
		v = vplus;

		if ( std::abs((L - Llast)/L) < std::get<0>(Tolerance) || std::sqrt(r2 / dot(b, b)) < std::get<1>(Tolerance) ){
			if (Verbose) std::printf("Tolerance met!\n");
			Sequence.push_back(std::make_tuple(v, Eigen::VectorXd::Zero(total_size)));
			return;
		}

		r -= alpha * Ap;
	}
	if (Verbose) std::printf("Maximal iterations!\n");
}

Eigen::VectorXd ConjugateGradient::Find(double R){
	for ( int i = 0; i < (int)Sequence.size(); i++ ){
		const auto& [v, ap] = Sequence[i];
		if ( dot(v + ap, v + ap) > R * R ){
			const double t = SteihaugToint(dot, v, ap, R);
			return v + t * ap;
		}
	}
	const auto& [v, ap] = Sequence.back();
	return v + ap;
}

#ifdef __PYTHON__
void Init_ConjugateGradient(pybind11::module_& m){
	pybind11::classh<ConjugateGradient, LinearSolver>(m, "ConjugateGradient")
		.def_readwrite("Sequence", &ConjugateGradient::Sequence)
		.def(pybind11::init<
			Iterate&, bool, bool, std::tuple<double, double>, int, bool
		>());
}
#endif

}
