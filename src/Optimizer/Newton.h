#include <array>

#include "../Manifold/Manifold.h"
#include "../LinearSolver/LinearSolver.h"

#include "TrustRegion.h"

namespace Maniverse{

void initLinearSolverForNewton(LinearSolver& ls, Iterate& M);

bool Newton(
		Iterate& M,
		TrustRegion& tr,
		LinearSolver& ls,
		std::array<double, 3> tol,
		int max_iter, int output
);

}
