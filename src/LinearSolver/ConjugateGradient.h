#pragma once

#include "LinearSolver.h"

namespace Maniverse{

class ConjugateGradient : public LinearSolver{ public:
	bool FrownNPC;
	ConjugateGradient(int FuncFreq, bool FrownNPC, std::tuple<double, double> Tolerance, bool Verbose): LinearSolver(FuncFreq, Tolerance, Verbose), FrownNPC(FrownNPC){};
	void Calculate(double R) override;
};

}
