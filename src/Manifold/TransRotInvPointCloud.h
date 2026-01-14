#pragma once

#include "Euclidean.h"

namespace Maniverse{

class TransRotInvPointCloud: public Euclidean{ public:
	TransRotInvPointCloud(EigenMatrix p, std::string geodesic = "EXACT");

	virtual int getDimension() const override;

	EigenMatrix InverseRetract(Manifold& N) const override;
	//EigenMatrix TransportTangent(EigenMatrix X, EigenMatrix Y) override const;
	EigenMatrix TransportManifold(EigenMatrix X, Manifold& N) const override;

	EigenMatrix TangentProjection(EigenMatrix A) const override;
	EigenMatrix TangentPurification(EigenMatrix A) const override;

	void setPoint(EigenMatrix p, bool purify) override;
	void getGradient() override;
	EigenMatrix getHessian(EigenMatrix HeX, EigenMatrix X, bool weingarten) const override;

	std::shared_ptr<Manifold> Share() const override;
};

}
