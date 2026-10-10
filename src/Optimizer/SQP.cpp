#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#endif

#include <Eigen/Dense>
#include <cmath>
#include <chrono>

#include "../Macro.h"
#include "../Manifold/Manifold.h"
#include "../LinearSolver/LinearSolver.h"

// https://link.springer.com/book/10.1007/978-0-387-40065-5

namespace Maniverse{

void initLinearSolverForNormal(LinearSolver& ls, Iterate& M){ // For the normal subproblem Eq 18.45
	if (ls.Verbose){
		std::printf("Manifold                                 : %s\n", M.getName().c_str());
		std::printf("Dimension number                         : %d\n", M.getDimension());
		std::printf("Extra constraint number                  : %d\n", (int)M.Constraints.size());
	}
	ls.dot = [&M](Eigen::VectorXd X, Eigen::VectorXd Y) -> double{ return M.Inner(X, Y); };
	ls.proj = [&M](Eigen::VectorXd X) -> Eigen::VectorXd{ return M.TangentProjection(X); };
	ls.A = [&M](Eigen::VectorXd X) -> Eigen::VectorXd{
		const int ncons = (int)M.Constraints.size();
		Eigen::VectorXd JtX = Eigen::VectorXd::Zero(ncons);
		for ( int i = 0; i < ncons; i++ ){
			JtX(i) = M.Inner(M.Constraints[i].Gradient, X);
		}
		Eigen::VectorXd JJtX = Eigen::VectorXd::Zero(M.Point.size());
		for ( int i = 0; i < ncons; i++ ){
			JJtX += M.Constraints[i].Gradient * JtX(i);
		}
		return JJtX;
	};
	ls.P = [](Eigen::VectorXd X) -> Eigen::VectorXd{ return X; };
	ls.b = Eigen::VectorXd::Zero(M.Point.size());
	for ( Constraint& c : M.Constraints ) ls.b -= c.Gradient * c.Func->Value;
}

void initLinearSolverForProjectedCG(LinearSolver& ls, Iterate& M){
}

#ifdef __PYTHON__
void Init_SQP(pybind11::module_& m){
	m.def("initLinearSolverForNormal", &initLinearSolverForNormal);
	m.def("initLinearSolverForProjectedCG", &initLinearSolverForProjectedCG);
}
#endif

}