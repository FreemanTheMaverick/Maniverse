#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#endif

#include <Eigen/Dense>
#include <array>
#include <cmath>

#include "../Manifold/Manifold.h"

#include "LinearSolver.h"

namespace Maniverse{

double SteihaugToint(
		std::function<double (Eigen::VectorXd, Eigen::VectorXd)> dot,
		Eigen::VectorXd v, Eigen::VectorXd p, double R){
	if ( p.norm() < 1e-15 ) return 0;
	const double A = dot(p, p);
	const double B = dot(v, p) * 2;
	const double C = dot(v, v) - R * R;
	const double t = ( std::sqrt( B * B - 4 * A * C ) - B ) / 2 / A;
	return t;
}

LinearSolver::LinearSolver(bool FrownNPC, std::array<double, 2> Tolerance, int MaxIter, bool Verbose) : FrownNPC(FrownNPC), Tolerance(Tolerance), MaxIter(MaxIter), Verbose(Verbose){
	if (Verbose){
		std::printf("Configuring linear solver for Newton step\n");
		std::printf("Linear solver type                       : %s\n", typeid(*this).name());
		std::printf("Frown at non-positive curvature          : %d\n", FrownNPC);
		std::printf("Tolerance of relative quadratic lowering : %E\n", Tolerance[0]);
		std::printf("Tolerance of relative residual           : %E\n", Tolerance[1]);
		std::printf("Maximal iterations                       : %d\n", MaxIter);
	}
}

#ifdef __PYTHON__
class PyLinearSolver : public LinearSolver, pybind11::trampoline_self_life_support{ public:
	using LinearSolver::LinearSolver;

	void Calculate(double R) override{
		PYBIND11_OVERRIDE_PURE(void, LinearSolver, Calculate, R);
	}

	Eigen::VectorXd Find(double R) override{
		PYBIND11_OVERRIDE_PURE(Eigen::VectorXd, LinearSolver, Find, R);
	}
};

void Init_LinearSolver(pybind11::module_& m){
	pybind11::classh<LinearSolver, PyLinearSolver>(m, "LinearSolver")
		.def_readwrite("dot", &LinearSolver::dot)
		.def_readwrite("proj", &LinearSolver::proj)
		.def_readwrite("A", &LinearSolver::A)
		.def_readwrite("b", &LinearSolver::b)
		.def_readwrite("P", &LinearSolver::P)
		.def_readwrite("FrownNPC", &LinearSolver::FrownNPC)
		.def_readwrite("Tolerance", &LinearSolver::Tolerance)
		.def_readwrite("MaxIter", &LinearSolver::MaxIter)
		.def_readwrite("Verbose", &LinearSolver::Verbose)
		.def(pybind11::init<bool, std::array<double, 2>, int, bool>())
		.def("Calculate", &LinearSolver::Calculate)
		.def("Find", &LinearSolver::Find);
}
#endif

}
