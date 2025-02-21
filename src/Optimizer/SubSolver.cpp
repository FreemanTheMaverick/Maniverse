#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#endif
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <tuple>
#include <functional>
#include <cstdio>
#include <chrono>
#include <cassert>
#include <string>
#include <tuple>
#include <memory>

#include "../Macro.h"
#include "../Manifold/Manifold.h"

#include <iostream>


std::vector<std::tuple<double, EigenMatrix, EigenMatrix>> TruncatedConjugateGradient(
		Manifold& M, double R,
		std::tuple<double, double, double> tol, int output){

	const double tol0 = std::get<0>(tol) * M.getDimension();
	const double tol1 = std::get<1>(tol) * M.P.size();
	const double tol2 = std::get<2>(tol) * M.P.size();
	if (output > 0){
		std::printf("Using truncated conjugated gradient optimizer on the tangent space of %s manifold\n", M.Name.c_str());
		std::printf("Convergence threshold:\n");
        std::printf("| Target change (T. C.)               : %E\n", tol0);
		std::printf("| Gradient norm (Grad.)               : %E\n", tol1);
		std::printf("| Independent variable update (V. U.) : %E\n", tol2);
		std::printf("| Itn. |       Target        |   T. C.  |  Grad.  |  V. U.  |  Time  |\n");
	}

	std::vector<std::tuple<double, EigenMatrix, EigenMatrix>> Vs; Vs.reserve(20);
	const double b2 = M.Inner(M.Gr, M.Gr);
	EigenMatrix v = EigenZero(M.Gr.rows(), M.Gr.cols());
	EigenMatrix r = -M.Gr;
	EigenMatrix p = -M.Gr;
	double vnorm = 0;
	double vplusnorm = 0;
	double r2 = b2;
	double L = 0;
	const auto start = __now__;

	EigenMatrix Hp = EigenZero(M.Gr.rows(), M.Gr.cols());
	EigenMatrix vplus = EigenZero(M.Gr.rows(), M.Gr.cols());

	for ( int iiter = 0; iiter < M.getDimension(); iiter++ ){
		if (output > 0) std::printf("| %4d |", iiter);
		Hp = M.TangentPurification(M.Hr(p));
		const double pHp = M.Inner(p, Hp);
		const double Llast = L;
		L = 0.5 * M.Inner(M.Hr(v), v) + M.Inner(M.Gr, v);
		const double deltaL = L - Llast;
		if (output > 0) std::printf("  %17.10f  | % 5.1E | %5.1E |", L, deltaL, std::sqrt(r2));

		const double alpha = r2 / pHp;
		vplus = M.TangentPurification(v + alpha * p);
		vplusnorm = std::sqrt(M.Inner(vplus, vplus));
		vnorm = std::sqrt(M.Inner(v, v));
		const double step = std::abs(alpha) * std::sqrt(M.Inner(p, p));
		if (output > 0) std::printf(" %5.1E | %6.3f |\n", step, __duration__(start, __now__));
		if (
			iiter > 0 && (
				(
					std::abs(deltaL) < tol0
					&& r2 < std::pow(tol1, 2)
					&& step < tol2
				) || std::abs(deltaL) < tol0 * tol0 /1000
			)
		){
			if (output > 0) std::printf("Tolerance met!\n");
			Vs.push_back(std::make_tuple(vnorm, v, p));
			return Vs;
		}

		if ( pHp <= 0 || vplusnorm >= R ){
			const double A = M.Inner(p, p);
			const double B = M.Inner(v, p) * 2.;
			const double C = vnorm * vnorm - R * R;
			const double t = ( std::sqrt( B * B - 4. * A * C ) - B ) / 2. / A;
			if (output > 0 && pHp <= 0) std::printf("Non-positive curvature!\n");
			if (output > 0 && vplusnorm >= R) std::printf("Out of trust region!\n");
			Vs.push_back(std::make_tuple(R, v + t * p, p));
			return Vs;
		}
		v = vplus;
		vnorm = vplusnorm;
		Vs.push_back(std::make_tuple(vnorm, v, p));
		const double r2old = r2;
		r = M.TangentPurification(r - alpha * Hp);
		r2 = M.Inner(r, r);
		const double beta = r2 / r2old;
		p = M.TangentPurification(r + beta * p);
	}
	if (output > 0) std::printf("Dimension completed!\n");
	return Vs;
}

EigenMatrix RestartTCG(Manifold& M, std::vector<std::tuple<double, EigenMatrix, EigenMatrix>>& Vs, double R){
	for ( int i = 0; i < (int)Vs.size(); i++ ) if ( std::get<0>(Vs[i]) > R ){
		const EigenMatrix v = std::get<1>(Vs[i]);
		const EigenMatrix p = std::get<2>(Vs[i]);
		const double A = M.Inner(p, p);
		const double B = M.Inner(v, p) * 2.;
		const double C = M.Inner(v, v) - R * R;
		const double t = ( std::sqrt( B * B - 4. * A * C ) - B ) / 2. / A;
		return v + t * p;
	}
	return std::get<1>(Vs.back());
}

#ifdef __PYTHON__
void Init_SubSolver(pybind11::module_& m){
	m.def("TruncatedConjugateGradient", &TruncatedConjugateGradient);
	m.def("RestartTCG", &RestartTCG);
}
#endif
