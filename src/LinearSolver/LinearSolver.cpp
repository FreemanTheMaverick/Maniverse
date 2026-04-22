#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#endif

#include <Eigen/Dense>
#include <tuple>
#include <cmath>

#include "../Manifold/Manifold.h"

#include "LinearSolver.h"

namespace Maniverse{

std::tuple<Eigen::VectorXd, Eigen::VectorXd> LinearSolver::SteihaugToint(Eigen::VectorXd v, Eigen::VectorXd p, double R){
	const double A = M->Inner(p, p);
	const double B = M->Inner(v, p) * 2.;
	const double C = M->Inner(v, v) - R * R;
	const double t = ( std::sqrt( B * B - 4. * A * C ) - B ) / 2. / A;
	return std::make_tuple(v + t * p, p);
}

Eigen::VectorXd LinearSolver::Find(double R){
	for ( int i = 0; i < (int)Sequence.size(); i++ ) if ( M->Inner(std::get<0>(Sequence[i]), std::get<0>(Sequence[i])) > R ){
		const auto& [v, p] = Sequence[i];
		const Eigen::VectorXd vnew = std::get<0>(SteihaugToint(v, p, R));
		return vnew;
	}
	return std::get<0>(this->Sequence.back());
}

#ifdef __PYTHON__
class PyLinearSolver : public LinearSolver, pybind11::trampoline_self_life_support{ public:
	using LinearSolver::LinearSolver;

	void Calculate(double R) override{
		PYBIND11_OVERRIDE_PURE(void, LinearSolver, Calculate, R);
	}
};

void Init_LinearSolver(pybind11::module_& m){
	pybind11::classh<LinearSolver, PyLinearSolver>(m, "LinearSolver")
		.def_readwrite("M", &LinearSolver::M)
		.def_readwrite("A", &LinearSolver::A)
		.def_readwrite("b", &LinearSolver::b)
		.def_readwrite("P", &LinearSolver::P)
		.def_readwrite("FuncFreq", &LinearSolver::FuncFreq)
		.def_readwrite("FrownNPC", &LinearSolver::FrownNPC)
		.def_readwrite("Verbose", &LinearSolver::Verbose)
		.def_readwrite("Tolerance", &LinearSolver::Tolerance)
		.def_readwrite("Sequence", &LinearSolver::Sequence)
		.def(pybind11::init<
				int, bool, std::tuple<double, double>, bool
		>()).def("SteihaugToint", &LinearSolver::SteihaugToint)
		.def("Calculate", &LinearSolver::Calculate)
		.def("Find", &LinearSolver::Find);
}
#endif

}
