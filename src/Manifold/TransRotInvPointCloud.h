#include "Euclidean.h"

class TransRotInvPointCloud: public Euclidean{ public:
	TransRotInvPointCloud(EigenMatrix p);

	virtual int getDimension() const override;

	EigenMatrix Logarithm(Manifold& N) const override;

	EigenMatrix TangentProjection(EigenMatrix A) const override;
	EigenMatrix TangentPurification(EigenMatrix A) const override;

	//EigenMatrix TransportTangent(EigenMatrix X, EigenMatrix Y) override const;
	EigenMatrix TransportManifold(EigenMatrix X, Manifold& N) const override;

	void setPoint(EigenMatrix p, bool purify) override;
	void getGradient() override;
	std::function<EigenMatrix (EigenMatrix)> getHessian(std::function<EigenMatrix (EigenMatrix)> He, bool weingarten) const override;

	std::unique_ptr<Manifold> Clone() const override;
};
