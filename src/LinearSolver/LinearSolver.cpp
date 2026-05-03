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

LinearSolver::LinearSolver(Iterate& M, bool constraint, bool FrownNPC, std::tuple<double, double> Tolerance, int MaxIter, bool Verbose) : FrownNPC(FrownNPC), Tolerance(Tolerance), MaxIter(MaxIter), Verbose(Verbose){
	if (Verbose){
		std::printf("Configuring linear solver for Newton step\n");
		std::printf("Manifold: %s\n", M.getName().c_str());
		std::printf("Dimension number: %d\n", M.getDimension());
		if (constraint) std::printf("Extra constraint: Yes\n");
		else std::printf("Extra constraint: No\n");
		std::printf("Frown at non-positive curvature          : %d\n", FrownNPC);
		std::printf("Tolerance of relative quadratic lowering : %E\n", std::get<0>(Tolerance));
		std::printf("Tolerance of relative residual           : %E\n", std::get<1>(Tolerance));
		std::printf("Maximal iterations                       : %d\n", MaxIter);
	}
	dot = [&M](Eigen::VectorXd X, Eigen::VectorXd Y) -> double{ return M.Inner(X, Y); };
	if (constraint){
		proj = [&M](Eigen::VectorXd X) -> Eigen::VectorXd{ return M.ConstraintProjection(M.TangentProjection(X)); };
		A = [&M](Eigen::VectorXd X) -> Eigen::VectorXd{ return M.ConstraintProjectedHessian(X); };
		P = [&M](Eigen::VectorXd X) -> Eigen::VectorXd{ return M.ConstraintProjectedPreconditioner(X); };
	}else{
		proj = [&M](Eigen::VectorXd X) -> Eigen::VectorXd{ return M.TangentProjection(X); };
		A = [&M](Eigen::VectorXd X) -> Eigen::VectorXd{ return M.Hessian(X); };
		P = [&M](Eigen::VectorXd X) -> Eigen::VectorXd{ return M.Preconditioner(X); };
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
		.def(pybind11::init<
			Iterate&, bool, bool, std::tuple<double, double>, int, bool
		>())
		.def("Calculate", &LinearSolver::Calculate)
		.def("Find", &LinearSolver::Find);
}
#endif

}
