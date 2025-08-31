#pragma once

#include "Manifold.h"

namespace Maniverse{

class Simplex: public Manifold{ public:
	Simplex(EigenMatrix p, std::string geodesic = "EXACT");

	int getDimension() const override;
	double Inner(EigenMatrix X, EigenMatrix Y) const override;

	EigenMatrix Retract(EigenMatrix X) const override;
	EigenMatrix InverseRetract(Manifold& N) const override;
	//EigenMatrix TransportTangent(EigenMatrix X, EigenMatrix Y) override;
	//EigenMatrix TransportManifold(EigenMatrix X, Manifold& N) override;

	EigenMatrix TangentProjection(EigenMatrix A) const override;
	EigenMatrix TangentPurification(EigenMatrix A) const override;

	void setPoint(EigenMatrix p, bool purify) override;
	void getGradient() override;
	std::function<EigenMatrix (EigenMatrix)> getHessian(std::function<EigenMatrix (EigenMatrix)> He, bool weingarten) const override;

	std::unique_ptr<Manifold> Clone() const override;
};

}
