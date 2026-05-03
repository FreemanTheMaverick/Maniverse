#include <Eigen/Dense>
#include <tuple>
#include <vector>
#include <functional>

namespace Maniverse{

std::tuple<std::vector<double>, std::vector<Eigen::VectorXd>> Lanczos(
		std::function<double (Eigen::VectorXd, Eigen::VectorXd)> dot,
		std::function<Eigen::VectorXd (Eigen::VectorXd)> proj,
		std::function<Eigen::VectorXd (Eigen::VectorXd)> A,
		Eigen::VectorXd b,
		std::function<Eigen::VectorXd (Eigen::VectorXd)> P,
		int m, bool output
);

std::tuple<std::vector<double>, std::vector<Eigen::VectorXd>> Lanczos(Iterate& M, int m, bool generalized, bool constraint, bool output);

}
