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
#include "SubSolver.h"

#include <iostream>


void TruncatedConjugateGradient::Run(){
	if (this->Verbose){
		std::printf("Using truncated conjugated gradient optimizer on the tangent space of %s manifold\n", this->M->Name.c_str());
		std::printf("| Itn. |       Target        |   T. C.  |  Grad.  |  V. U.  |  Time  |\n");
	}

	this->Sequence.clear(); this->Sequence.reserve(20);
	const double b2 = this->M->Inner(this->M->Gr, this->M->Gr);
	EigenMatrix v = EigenZero(this->M->Gr.rows(), this->M->Gr.cols());
	EigenMatrix r = - this->M->Gr;
	EigenMatrix p = - this->M->Gr;
	double vnorm = 0;
	double vplusnorm = 0;
	double r2 = b2;
	double L = 0;
	const auto start = __now__;

	EigenMatrix Hp = EigenZero(this->M->Gr.rows(), this->M->Gr.cols());
	EigenMatrix vplus = EigenZero(this->M->Gr.rows(), this->M->Gr.cols());

	for ( int iiter = 0; iiter < this->M->getDimension(); iiter++ ){
		if (this->Verbose) std::printf("| %4d |", iiter);
		Hp = this->M->TangentPurification((*(this->Func))(p));
		const double pHp = this->M->Inner(p, Hp);
		const double Llast = L;
		if (this->ShowTarget) L = 0.5 * this->M->Inner((*(this->Func))(v), v) + this->M->Inner(this->M->Gr, v);
		else L = std::nan("");
		const double deltaL = L - Llast;
		if (this->Verbose) std::printf("  %17.10f  | % 5.1E | %5.1E |", L, deltaL, std::sqrt(r2));

		const double alpha = r2 / pHp;
		vplus = this->M->TangentPurification(v + alpha * p);
		vplusnorm = std::sqrt(this->M->Inner(vplus, vplus));
		vnorm = std::sqrt(this->M->Inner(v, v));
		const double step = std::abs(alpha) * std::sqrt(this->M->Inner(p, p));
		if (this->Verbose) std::printf(" %5.1E | %6.3f |\n", step, __duration__(start, __now__));
		if ( iiter > 0 && ( this->Tolerance(deltaL, L, std::sqrt(r2), step) ) ){
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
		r = this->M->TangentPurification(r - alpha * Hp);
		r2 = this->M->Inner(r, r);
		const double beta = r2 / r2old;
		p = this->M->TangentPurification(r + beta * p);
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

#ifdef __PYTHON__
void Init_SubSolver(pybind11::module_& m){
	pybind11::class_<TruncatedConjugateGradient>(m, "TruncatedConjugateGradient")
		.def_readwrite("M", &TruncatedConjugateGradient::M)
		.def_readwrite("Func", &TruncatedConjugateGradient::Func)
		.def_readwrite("Verbose", &TruncatedConjugateGradient::Verbose)
		.def_readwrite("ShowTarget", &TruncatedConjugateGradient::ShowTarget)
		.def_readwrite("Radius", &TruncatedConjugateGradient::Radius)
		.def_readwrite("Tolerance", &TruncatedConjugateGradient::Tolerance)
		.def_readwrite("Sequence", &TruncatedConjugateGradient::Sequence)
		.def(pybind11::init<>())
		.def(pybind11::init<Manifold*, std::function<EigenMatrix (EigenMatrix)>*, bool, bool>())
		.def("Run", &TruncatedConjugateGradient::Run)
		.def("Find", &TruncatedConjugateGradient::Find);
}
#endif
