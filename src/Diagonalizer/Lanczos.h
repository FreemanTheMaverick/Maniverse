#include <Eigen/Dense>
#include <tuple>
#include <vector>

namespace Maniverse{

std::tuple<std::vector<double>, std::vector<Eigen::VectorXd>> Lanczos(Iterate& M, int m, double beta_min, int output);

}
