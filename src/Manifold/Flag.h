#pragma once

#include <Eigen/Dense>
#include <vector>
#include <tuple>
#include <string>
#include <memory>

#include "Stiefel.h"

namespace Maniverse{

class Flag: public Stiefel{ public:
	std::vector<std::tuple<int, int>> BlockParameters;
	void setBlockParameters(std::vector<int>);

	Flag(Eigen::MatrixXd p, std::string geodesic = "POLAR");

	int getDimension() const override;

	Eigen::MatrixXd TangentProjection(Eigen::MatrixXd A) const override;
	Eigen::MatrixXd TangentPurification(Eigen::MatrixXd A) const override;

	Eigen::MatrixXd getHessian(Eigen::MatrixXd HeX, Eigen::MatrixXd X, bool weingarten) const override;

	std::unique_ptr<Manifold> Clone() const override;
	std::shared_ptr<Manifold> Share() const override;
};

}

#define FlagGetColumns(big_mat, imat)\
	big_mat( Eigen::placeholders::all, Eigen::seqN(\
			std::get<0>(BlockParameters[imat]),\
			std::get<1>(BlockParameters[imat])\
	) )

#define FlagGetBlock(big_mat, imat, jmat)\
	big_mat.block(\
			std::get<0>(BlockParameters[imat]),\
			std::get<0>(BlockParameters[jmat]),\
			std::get<1>(BlockParameters[imat]),\
			std::get<1>(BlockParameters[jmat])\
	)
