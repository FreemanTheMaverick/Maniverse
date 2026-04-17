#pragma once

#include <Eigen/Dense>
#include <string>
#include <memory>

#include "Manifold.h"

namespace Maniverse{

class Simplex: public Manifold{ public:
	Simplex(Eigen::MatrixXd p, std::string geodesic = "EXACT");

	int getDimension() const override;
	double Inner(Eigen::MatrixXd X, Eigen::MatrixXd Y) const override;

	Eigen::MatrixXd Retract(Eigen::MatrixXd X) const override;
	Eigen::MatrixXd InverseRetract(Manifold& N) const override;

	Eigen::MatrixXd TangentProjection(Eigen::MatrixXd A) const override;
	Eigen::MatrixXd TangentPurification(Eigen::MatrixXd A) const override;

	void setPoint(Eigen::MatrixXd p, bool purify) override;
	void getGradient() override;
	Eigen::MatrixXd getHessian(Eigen::MatrixXd HeX, Eigen::MatrixXd X, bool weingarten) const override;

	std::unique_ptr<Manifold> Clone() const override;
	std::shared_ptr<Manifold> Share() const override;
};

}
