#include <Eigen/Dense>
#include <functional>
#include <vector>
#include <tuple>

#include "../Manifold/Manifold.h"
#include "../LinearSolver/LinearSolver.h"

#include "TrustRegion.h"

namespace Maniverse{

bool Newton(
		Iterate& M,
		TrustRegion& tr,
		LinearSolver& ls,
		std::tuple<double, double, double> tol,
		int max_iter, int output
);

}
