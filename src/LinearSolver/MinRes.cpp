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

static std::tuple<Eigen::VectorXd, Eigen::VectorXd> SteihaugToint(
		std::function<double (Eigen::VectorXd, Eigen::VectorXd)> dot,
		Eigen::VectorXd v, Eigen::VectorXd p, double R){
	const double A = dot(p, p);
	const double B = dot(v, p) * 2;
	const double C = dot(v, v) - R * R;
	const double t = ( std::sqrt( B * B - 4 * A * C ) - B ) / 2 / A;
	return std::make_tuple(v + t * p, p);
}

void MinRes::Calculate(double R){
	if (Verbose){
		std::printf("Linear system solving with MinRes\n");
		std::printf("Maximal search radius          : %f\n", R);
		std::printf("Maximal number of iterations   : %d\n", MaxIter);
		std::printf("| Itn. |  Resi.  |       Target        |   Diff.  | Length  |  Time  |\n");
	}

	const int total_size = b.size();
	Sequence.clear(); Sequence.reserve(20);

	double beta = std::sqrt(dot(b, b));
	Eigen::VectorXd r_m1 = b;
	Eigen::VectorXd v = b / beta;
	Eigen::VectorXd v_m1(total_size);
	Eigen::VectorXd x_m1 = v_m1, d_m1 = v_m1, d_m2 = v_m1;
	double c_m1 = -1;
	double s_m1 = 0;
	double phi_m1 = beta, tau_m1 = beta;
	double delta1 = 0;
	double epsilon = 0;

	double L = 0;
	const auto start = __now__;

	for ( int iiter = 0; iiter < MaxIter; iiter++ ){
		if (Verbose) std::printf("| %4d ", iiter);

		const double r2 = dot(r_m1, r_m1);
		if (Verbose) std::printf("| %5.1E ", std::sqrt(r2));

		const double Llast = L;
		L = 0.5 * dot( - r_m1 - b, x_m1 );
		if (Verbose) std::printf("|  %17.10f  | % 5.1E |", L, L - Llast);

		const double xnorm = std::sqrt(dot(x_m1, x_m1));
		if (Verbose) std::printf(" %5.1E | %6.3f |\n", xnorm, __duration__(start, __now__));

		if ( std::abs((L - Llast)/L) < std::get<0>(Tolerance) || std::sqrt(r2 / dot(b, b)) < std::get<1>(Tolerance) ){
			if (Verbose) std::printf("Tolerance met!\n");
			Sequence.push_back(std::make_tuple(x_m1, P(d_m2)));
			return;
		}

		Eigen::VectorXd p = proj(A(P(v)));
		const double alpha = dot(v, p);
		p -= beta * v_m1 + alpha * v;
		const double beta_p1 = std::sqrt(dot(p, p));
		const double delta2 = c_m1 * delta1 + s_m1 * alpha;
		const double gamma1 = s_m1 * delta1 - c_m1 * alpha;
		const double epsilon_p1 = s_m1 * beta_p1;
		const double delta1_p1 = - c_m1 * beta_p1;

		const double gamma2 = std::hypot(gamma1, beta_p1);
		if ( gamma2 > 1e-12 ){
			const double c = gamma1 / gamma2;
			const double s = beta_p1 / gamma2;
			const double tau = c * phi_m1;
			const double phi = s * phi_m1;
			const Eigen::VectorXd d = ( v - delta2 * d_m1 - epsilon * d_m2 ) / gamma2;
			const Eigen::VectorXd x = x_m1 + tau * P(d);
			const double xplusnorm = std::sqrt(dot(x, x));
			if ( ( FrownNPC && c_m1 * gamma1 >= 0 ) || xplusnorm >= R ){
				if ( Verbose && FrownNPC && c_m1 * gamma1 >= 0 ) std::printf("Non-positive curvature!\n");
				if ( Verbose && xplusnorm >= R ) std::printf("Out of trust region!\n");
				Sequence.push_back(SteihaugToint(dot, x_m1, P(d), R));
				return;
			}
			if ( std::abs(beta_p1) > 1e-16 ){
				const Eigen::VectorXd v_p1 = p / beta_p1;
				const Eigen::VectorXd r = s * s * r_m1 - phi * c * v_p1;
				r_m1 = r;
				v_m1 = v; v = v_p1;
				x_m1 = x;
				d_m2 = d_m1; d_m1 = d;
				c_m1 = c;
				s_m1 = s;
				phi_m1 = phi;
				tau_m1 = tau;
				beta = beta_p1;
				epsilon = epsilon_p1;
				delta1 = delta1_p1;
			}else{
				if (Verbose) std::printf("Early stop due to the small Beta!\n");
				return;
			}
		}else{
			if (Verbose) std::printf("Early stop due to the small Gamma(2)!\n");
			return;
		}
	}
	if (Verbose) std::printf("Maximal iterations!\n");
}

Eigen::VectorXd MinRes::Find(double R){
	for ( int i = 0; i < (int)Sequence.size(); i++ ) if ( dot(std::get<0>(Sequence[i]), std::get<0>(Sequence[i])) > R ){
		const auto& [v, p] = Sequence[i];
		const Eigen::VectorXd vnew = std::get<0>(SteihaugToint(dot, v, p, R));
		return vnew;
	}
	return std::get<0>(this->Sequence.back());
}

#ifdef __PYTHON__
void Init_MinRes(pybind11::module_& m){
	pybind11::classh<MinRes, LinearSolver>(m, "MinRes")
		.def_readwrite("Sequence", &MinRes::Sequence)
		.def(pybind11::init<
			Iterate&, bool, bool, std::tuple<double, double>, int, bool
		>());
}
#endif

}
