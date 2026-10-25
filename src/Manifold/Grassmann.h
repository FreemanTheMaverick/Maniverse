#pragma once

#include <Eigen/Dense>
#include <tuple>
#include <string>
#include <memory>

#include "Manifold.h"

namespace Maniverse{

class Grassmann: public Manifold{ public:
	Eigen::MatrixXd Projector;
	mutable std::vector<std::tuple<Eigen::MatrixXd, Eigen::MatrixXd>> LogCache;
	mutable std::vector<std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>> TransportTangentCache;

	Grassmann(Eigen::MatrixXd p, std::string geodesic = "EXACT");

	int getDimension() const override;
	double Inner(Eigen::MatrixXd X, Eigen::MatrixXd Y) const override;

	Eigen::MatrixXd Retract(Eigen::MatrixXd X) const override;
	Eigen::MatrixXd InverseRetract(Manifold& N) const override;

	Eigen::MatrixXd TangentProjection(Eigen::MatrixXd A) const override;
	Eigen::MatrixXd TangentPurification(Eigen::MatrixXd A) const override;

	Eigen::MatrixXd TransportTangent(Eigen::MatrixXd X, Eigen::MatrixXd Y) const override;
	Eigen::MatrixXd TransportManifold(Eigen::MatrixXd X, Manifold& N) const override;

	void setPoint(Eigen::MatrixXd p, bool purify) override;

	void getGradient() override;
	Eigen::MatrixXd getHessian(Eigen::MatrixXd HeX, Eigen::MatrixXd X, bool weingarten) const override;

	std::unique_ptr<Manifold> Clone() const override;
	std::shared_ptr<Manifold> Share() const override;
};

}
