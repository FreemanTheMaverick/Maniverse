#pragma once

#include <Eigen/Dense>
#include <string>
#include <memory>

#include "Stiefel.h"

namespace Maniverse{

class Orthogonal: public Stiefel{ public:
	Orthogonal(Eigen::MatrixXd p, std::string geodesic = "POLAR");

	Eigen::MatrixXd Retract(Eigen::MatrixXd X) const override;
	Eigen::MatrixXd InverseRetract(Manifold& N) const override;

	std::unique_ptr<Manifold> Clone() const override;
	std::shared_ptr<Manifold> Share() const override;
};

}
