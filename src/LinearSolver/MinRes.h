#pragma once

#include "LinearSolver.h"

namespace Maniverse{

class MinRes : public LinearSolver{ public:
	MinRes(int FuncFreq, std::tuple<double, double> Tolerance, bool Verbose): LinearSolver(FuncFreq, Tolerance, Verbose){};
	void Calculate(double R) override;
};

}
