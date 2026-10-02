#include <array>

#include "../Manifold/Manifold.h"

namespace Maniverse{

bool LBFGS(
		Iterate& M,
		std::array<double, 3> tol,
		int max_mem, int max_iter,
		double c1, double tau, int ls_max_iter,
		int output
);

}
