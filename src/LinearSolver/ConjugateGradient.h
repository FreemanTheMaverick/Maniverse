#pragma once

#include "LinearSolver.h"

namespace Maniverse{

class ConjugateGradient : public LinearSolver{ public:
	std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd>> Sequence;
	ConjugateGradient(bool FrowNPC, std::array<double, 2> Tolerance, int MaxIter, bool Verbose);
	void Calculate(double R) override;
	Eigen::VectorXd Find(double R) override;
};

}
