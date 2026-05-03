#pragma once

#include "LinearSolver.h"

namespace Maniverse{

class MinRes : public LinearSolver{ public:
	std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd>> Sequence;
	using LinearSolver::LinearSolver;
	void Calculate(double R) override;
	Eigen::VectorXd Find(double R) override;
};

}
