#include "Manifold.h"

class Euclidean: public Manifold{ public:
	Euclidean(EigenMatrix p);

	virtual int getDimension() const override;
	double Inner(EigenMatrix X, EigenMatrix Y) const override;

	EigenMatrix Exponential(EigenMatrix X) const override;
	virtual EigenMatrix Logarithm(Manifold& N) const override;

	virtual EigenMatrix TangentProjection(EigenMatrix A) const override;
	virtual EigenMatrix TangentPurification(EigenMatrix A) const override;

	//EigenMatrix TransportTangent(EigenMatrix X, EigenMatrix Y) override;
	//EigenMatrix TransportManifold(EigenMatrix X, Manifold& N) override;

	virtual void setPoint(EigenMatrix p, bool purify) override;
	virtual void getGradient() override;
	virtual std::function<EigenMatrix (EigenMatrix)> getHessian(std::function<EigenMatrix (EigenMatrix)> h, bool weingarten) const override;

	virtual std::unique_ptr<Manifold> Clone() const override;
};
