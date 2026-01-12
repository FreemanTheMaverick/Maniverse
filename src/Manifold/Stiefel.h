#pragma once

#include "Manifold.h"

namespace Maniverse{

class Stiefel: public Manifold{ public:
	Stiefel(EigenMatrix p, std::string geodesic = "POLAR");

	virtual int getDimension() const override;
	double Inner(EigenMatrix X, EigenMatrix Y) const override;

	virtual EigenMatrix Retract(EigenMatrix X) const override;
	virtual EigenMatrix InverseRetract(Manifold& N) const override;
	virtual EigenMatrix TransportTangent(EigenMatrix Y, EigenMatrix Z) const override;
	virtual EigenMatrix TransportManifold(EigenMatrix X, Manifold& N) const override;

	virtual EigenMatrix TangentProjection(EigenMatrix X) const override;
	virtual EigenMatrix TangentPurification(EigenMatrix X) const override;

	virtual void setPoint(EigenMatrix p, bool purify) override;

	virtual void getGradient() override;
	virtual EigenMatrix getHessian(EigenMatrix HeX, EigenMatrix X, bool weingarten) const override;
	std::unique_ptr<Manifold> Clone() const override;
};

}
