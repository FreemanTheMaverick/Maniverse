#include "Manifold.h"

class Stiefel: public Manifold{ public:
	Stiefel(EigenMatrix p);

	virtual int getDimension() const override;
	double Inner(EigenMatrix X, EigenMatrix Y) const override;

	virtual EigenMatrix Exponential(EigenMatrix X) const override;

	virtual EigenMatrix TangentProjection(EigenMatrix X) const override;
	virtual EigenMatrix TangentPurification(EigenMatrix X) const override;

	virtual void setPoint(EigenMatrix p, bool purify) override;

	virtual void getGradient() override;
	virtual std::function<EigenMatrix (EigenMatrix)> getHessian(std::function<EigenMatrix (EigenMatrix)> h, bool weingarten) const override;
	std::unique_ptr<Manifold> Clone() const override;
};
