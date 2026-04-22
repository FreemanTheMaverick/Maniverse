#pragma once

#include <Eigen/Dense>
#include <functional>
#include <tuple>
#include <vector>
#include <cstdio>

#include "../Manifold/Manifold.h"

namespace Maniverse{

class LinearSolver{ public:
	Iterate* M;
	std::function<Eigen::VectorXd (Eigen::VectorXd)> A;
	Eigen::VectorXd b;
	std::function<Eigen::VectorXd (Eigen::VectorXd)> P;
	int FuncFreq;
	bool FrownNPC;
	std::tuple<double, double> Tolerance;
	bool Verbose;
	std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd>> Sequence;

	LinearSolver(
			int FuncFreq,
			bool FrownNPC,
			std::tuple<double, double> Tolerance,
			bool Verbose
	):
		FuncFreq(FuncFreq), FrownNPC(FrownNPC),
		Tolerance(Tolerance), Verbose(Verbose)
	{};

	std::tuple<Eigen::VectorXd, Eigen::VectorXd> SteihaugToint(Eigen::VectorXd v, Eigen::VectorXd p, double R);
	virtual void Calculate(double R) = 0;
	Eigen::VectorXd Find(double R);
};

}
