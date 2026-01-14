#pragma once

#include "Stiefel.h"

namespace Maniverse{

class Flag: public Stiefel{ public:
	std::vector<std::tuple<int, int>> BlockParameters;
	void setBlockParameters(std::vector<int>);

	Flag(EigenMatrix p, std::string geodesic = "POLAR");

	int getDimension() const override;

	EigenMatrix TangentProjection(EigenMatrix A) const override;
	EigenMatrix TangentPurification(EigenMatrix A) const override;

	EigenMatrix getHessian(EigenMatrix HeX, EigenMatrix X, bool weingarten) const override;

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
