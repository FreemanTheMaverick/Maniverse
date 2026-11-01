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

	Flag(Eigen::MatrixXd p, std::string geodesic = "POLAR") : Stiefel(p, geodesic){};
	Flag(Eigen::MatrixXd p, Eigen::MatrixXd B, std::string geodesic = "POLAR") : Stiefel(p, B, geodesic){};

	int getDimension() const override;

	Eigen::MatrixXd symf(Eigen::MatrixXd PtX) const override;

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
