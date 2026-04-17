#pragma once

#include <Eigen/Dense>
#include <string>
#include <memory>

#include "Euclidean.h"

namespace Maniverse{

class RealSymmetric: public Euclidean{ public:
	RealSymmetric(Eigen::MatrixXd p, std::string geodesic = "EXACT");

	int getDimension() const override;

	Eigen::MatrixXd TangentProjection(Eigen::MatrixXd A) const override;
	Eigen::MatrixXd TangentPurification(Eigen::MatrixXd A) const override;

	void setPoint(Eigen::MatrixXd p, bool purify) override;

	void getGradient() override;
	Eigen::MatrixXd getHessian(Eigen::MatrixXd HeX, Eigen::MatrixXd X, bool weingarten) const override;

	std::unique_ptr<Manifold> Clone() const override;
	std::shared_ptr<Manifold> Share() const override;
};

}
