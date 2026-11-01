#include <Maniverse/Manifold/Flag.h>
#include <Maniverse/LinearSolver/ConjugateGradient.h>
#include <Maniverse/Optimizer/Newton.h>
#include <Eigen/Dense>
#include <algorithm>
#include <cstdio>
#include <vector>

namespace mv = Maniverse;

class PrincipalSubspace : public mv::Function {
public:
    Eigen::MatrixXd A;

	PrincipalSubspace() : A(4, 4) {
		A << 4.0, 1.0, 0.5, 0.0,
		     1.0, 3.0, 0.0, 0.5,
		     0.5, 0.0, 2.0, 0.5,
		     0.0, 0.5, 0.5, 1.0;
	}

	void Calculate(std::vector<Eigen::MatrixXd> points,
	               std::vector<int> derivatives) override {
		const Eigen::MatrixXd &C = points[0];
		if (std::count(derivatives.begin(), derivatives.end(), 0))
			Value = -C.cwiseProduct(A * C).sum();
		if (std::count(derivatives.begin(), derivatives.end(), 1))
			Gradient = {-2.0 * A * C};
	}

	std::vector<Eigen::MatrixXd> Hessian(
			std::vector<Eigen::MatrixXd> directions) const override {
		return {-2.0 * A * directions[0]};
	}
};

int main() {
	Eigen::MatrixXd C0 = Eigen::MatrixXd::Identity(4, 2);
	mv::Flag manifold(C0);
	manifold.setBlockParameters({2});
	PrincipalSubspace objective;
	mv::Iterate M(objective, {manifold.Share()});

	mv::TrustRegion tr;
	mv::ConjugateGradient cg(1, {1e-4, 1e-4}, M.getDimension(), 0);
	mv::initLinearSolverForNewton(cg, M);
	const bool converged = mv::Newton(
		M, tr, cg, {1e-6, 1e-6, 1e-6}, 20, 1
	);

	const Eigen::MatrixXd C = M.getPoint()[0];
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(objective.A);
	const Eigen::MatrixXd C_exact = eig.eigenvectors().rightCols(2);
	const double orthogonality_error =
		(C.transpose() * C - Eigen::MatrixXd::Identity(2, 2)).norm();
	const double projector_error =
		(C * C.transpose() - C_exact * C_exact.transpose()).norm();

	if (converged && orthogonality_error < 1e-6 && projector_error < 1e-5) {
		std::printf("Success\n");
		return 0;
	}

	std::printf(
		"Failed: converged=%d, orthogonality error=%.3e, projector error=%.3e\n",
		converged, orthogonality_error, projector_error
	);
	return 0;
}
