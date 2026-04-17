#pragma once

#include <Eigen/Dense>
#include <string>
#include <memory>

#include "Manifold.h"

namespace Maniverse{

class Euclidean: public Manifold{ public:
	Euclidean(Eigen::MatrixXd p, std::string geodesic = "EXACT");

	virtual int getDimension() const override;
	double Inner(Eigen::MatrixXd X, Eigen::MatrixXd Y) const override;

	Eigen::MatrixXd Retract(Eigen::MatrixXd X) const override;
	virtual Eigen::MatrixXd InverseRetract(Manifold& N) const override;

	virtual Eigen::MatrixXd TangentProjection(Eigen::MatrixXd A) const override;
	virtual Eigen::MatrixXd TangentPurification(Eigen::MatrixXd A) const override;

	Eigen::MatrixXd TransportTangent(Eigen::MatrixXd X, Eigen::MatrixXd Y) const override;
	Eigen::MatrixXd TransportManifold(Eigen::MatrixXd X, Manifold& N) const override;

	virtual void setPoint(Eigen::MatrixXd p, bool purify) override;
	virtual void getGradient() override;
	virtual Eigen::MatrixXd getHessian(Eigen::MatrixXd HeX, Eigen::MatrixXd X, bool weingarten) const override;

	virtual std::unique_ptr<Manifold> Clone() const override;
	virtual std::shared_ptr<Manifold> Share() const override;
};

}
