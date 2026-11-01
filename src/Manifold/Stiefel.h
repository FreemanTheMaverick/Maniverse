#pragma once

#include <Eigen/Dense>
#include <string>
#include <memory>

#include "Manifold.h"

namespace Maniverse{

class Stiefel: public Manifold{ public:
	Eigen::MatrixXd B;
	Eigen::MatrixXd Binv;
	Eigen::MatrixXd Bsqrt;

	Stiefel(Eigen::MatrixXd p, std::string geodesic = "POLAR");
	Stiefel(Eigen::MatrixXd p, Eigen::MatrixXd B, std::string geodesic = "POLAR");

	virtual int getDimension() const override;
	double Inner(Eigen::MatrixXd X, Eigen::MatrixXd Y) const override;
	Eigen::MatrixXd Flat(Eigen::MatrixXd X) const override;
	Eigen::MatrixXd Sharp(Eigen::MatrixXd X) const override;

	virtual Eigen::MatrixXd Retract(Eigen::MatrixXd X) const override;
	virtual Eigen::MatrixXd InverseRetract(Manifold& N) const override;
	virtual Eigen::MatrixXd TransportTangent(Eigen::MatrixXd Y, Eigen::MatrixXd Z) const override;
	virtual Eigen::MatrixXd TransportManifold(Eigen::MatrixXd X, Manifold& N) const override;

	virtual Eigen::MatrixXd symf(Eigen::MatrixXd PtBX) const;
	Eigen::MatrixXd TangentProjection(Eigen::MatrixXd X) const override;
	Eigen::MatrixXd TangentPurification(Eigen::MatrixXd X) const override;

	void setPoint(Eigen::MatrixXd p, bool purify) override;

	void getGradient() override;
	Eigen::MatrixXd getHessian(Eigen::MatrixXd HeX, Eigen::MatrixXd X) const override;

	std::unique_ptr<Manifold> Clone() const override;
	std::shared_ptr<Manifold> Share() const override;
};

}
