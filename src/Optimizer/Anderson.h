#include <array>

#include "../Manifold/Manifold.h"

namespace Maniverse{

bool Anderson(
		Iterate& M,
		std::array<double, 3> tol,
		double beta, int max_mem, int max_iter,
		int output
);

}
