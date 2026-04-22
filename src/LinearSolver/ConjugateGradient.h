#pragma once

#include "LinearSolver.h"

namespace Maniverse{

class ConjugateGradient : public LinearSolver{ public:
	using LinearSolver::LinearSolver;
	void Calculate(double R) override;
};

}
