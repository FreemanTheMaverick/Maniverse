#include <Eigen/Dense>

#include "../Manifold/Manifold.h"

namespace Maniverse{

bool ArmijoBacktracking(
		Iterate& M, Eigen::MatrixXd& S,
		double c1, double tau, int max_iter,
		int output
);

}
