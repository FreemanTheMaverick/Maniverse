#pragma once

#include <Eigen/Dense>
#include <functional>
#include <tuple>
#include <vector>
#include <cstdio>

#include "../Manifold/Manifold.h"

namespace Maniverse{

class LinearSolver{ public:
	std::function<double (Eigen::VectorXd, Eigen::VectorXd)> dot;
	std::function<Eigen::VectorXd (Eigen::VectorXd)> proj;
	std::function<Eigen::VectorXd (Eigen::VectorXd)> A;
	Eigen::VectorXd b;
	std::function<Eigen::VectorXd (Eigen::VectorXd)> P;
	bool FrownNPC;
	std::tuple<double, double> Tolerance;
	int MaxIter;
	bool Verbose;

	LinearSolver(Iterate& M, bool Constraint, bool FrownNPC, std::tuple<double, double> Tolerance, int MaxIter, bool Verbose);

	virtual void Calculate(double R) = 0;
	virtual Eigen::VectorXd Find(double R) = 0;
};

}
