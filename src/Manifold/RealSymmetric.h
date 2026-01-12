#pragma once

#include "Euclidean.h"

namespace Maniverse{

class RealSymmetric: public Euclidean{ public:
	RealSymmetric(EigenMatrix p, std::string geodesic = "EXACT");

	int getDimension() const override;

	EigenMatrix TangentProjection(EigenMatrix A) const override;
	EigenMatrix TangentPurification(EigenMatrix A) const override;

	void setPoint(EigenMatrix p, bool purify) override;

	void getGradient() override;
	EigenMatrix getHessian(EigenMatrix HeX, EigenMatrix X, bool weingarten) const override;

	std::unique_ptr<Manifold> Clone() const override;
};

}
