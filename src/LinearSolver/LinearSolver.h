#pragma once

#include <Eigen/Dense>
#include <functional>
#include <array>
#include <vector>
#include <cstdio>

#include "../Manifold/Manifold.h"

namespace Maniverse{

double SteihaugToint(
		std::function<double (Eigen::VectorXd, Eigen::VectorXd)> dot,
		Eigen::VectorXd v, Eigen::VectorXd p, double R
);

class LinearSolver{ public:
	std::function<double (Eigen::VectorXd, Eigen::VectorXd)> dot;
	std::function<Eigen::VectorXd (Eigen::VectorXd)> proj;
	std::function<Eigen::VectorXd (Eigen::VectorXd)> A;
	Eigen::VectorXd b;
	std::function<Eigen::VectorXd (Eigen::VectorXd)> P;
	bool FrownNPC;
	std::array<double, 2> Tolerance;
	int MaxIter;
	bool Verbose;

	LinearSolver(bool FrownNPC, std::array<double, 2> Tolerance, int MaxIter, bool Verbose);
	virtual void Calculate(double R) = 0;
	virtual Eigen::VectorXd Find(double R) = 0;
};

}
