#pragma once

#include <Eigen/Dense>
#include <string>
#include <memory>

#include "Manifold.h"

namespace Maniverse{

class Stiefel: public Manifold{ public:
	Stiefel(Eigen::MatrixXd p, std::string geodesic = "POLAR");

	virtual int getDimension() const override;
	double Inner(Eigen::MatrixXd X, Eigen::MatrixXd Y) const override;

	virtual Eigen::MatrixXd Retract(Eigen::MatrixXd X) const override;
	virtual Eigen::MatrixXd InverseRetract(Manifold& N) const override;
	virtual Eigen::MatrixXd TransportTangent(Eigen::MatrixXd Y, Eigen::MatrixXd Z) const override;
	virtual Eigen::MatrixXd TransportManifold(Eigen::MatrixXd X, Manifold& N) const override;

	virtual Eigen::MatrixXd TangentProjection(Eigen::MatrixXd X) const override;
	virtual Eigen::MatrixXd TangentPurification(Eigen::MatrixXd X) const override;

	virtual void setPoint(Eigen::MatrixXd p, bool purify) override;

	virtual void getGradient() override;
	virtual Eigen::MatrixXd getHessian(Eigen::MatrixXd HeX, Eigen::MatrixXd X, bool weingarten) const override;

	std::unique_ptr<Manifold> Clone() const override;
	std::shared_ptr<Manifold> Share() const override;
};

}
